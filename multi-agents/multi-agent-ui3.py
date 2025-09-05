import logging
from logging.handlers import RotatingFileHandler
import traceback
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from phi.model.openai import OpenAIChat
import gradio as gr
import autogen
from typing import Iterator
from phi.utils.pprint import pprint_run_response
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time
import tiktoken  # For token counting

# Add "use_docker": False in autogen configuration
autogen_config = {"use_docker": False}

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PHI_API_KEY = os.getenv("PHI_API_KEY")
openai.api_key = os.getenv("PHI_API_KEY")
OpenAIChat.api = os.getenv("PHI_API_KEY")

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'agent_chat.log'

file_handler = RotatingFileHandler(
    log_file, mode='a', maxBytes=1 * 1024 * 1024, backupCount=3, encoding='utf-8', delay=False
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

logging.info("Application starting...")

# Database setup
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_input = Column(Text)
    agent_name = Column(String)
    agent_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

try:
    engine = create_engine("sqlite:///chat_history.db")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logging.info("Database connection established")
except Exception as e:
    logging.error(f"Database connection failed: {str(e)}", exc_info=True)
    raise

# Token counting setup
encoding = tiktoken.encoding_for_model("gpt-4")  # Use GPT-4 tokenizer for counting

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens."""
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

# Rate limiting and retry mechanism
class RateLimiter:
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def execute_with_retry(self, func, *args, **kwargs):
        retries = 0
        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                    retries += 1
                    wait_time = self.backoff_factor ** retries
                    logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("Max retries exceeded. Rate limit still exceeded.")

rate_limiter = RateLimiter(max_retries=3, backoff_factor=2)

# Topic Continuity Detection Prompt
topic_continuity_prompt = """Analyze if the new input continues the previous context. Consider:
1. Direct references to previous content/entities
2. Continuation of the same task type
3. Use of context-specific terms
4. Temporal relevance

Respond ONLY with 'continue' if the input continues the topic, otherwise 'new'.

Previous Context ({agent_name}):
{previous_context}

New Input:
{user_input}

Response: """

def check_topic_continuity(user_input: str, previous_context: str, agent_name: str) -> bool:
    """Enhanced continuity check with keyword validation"""
    if not previous_context or previous_context == "None":
        return False
    
    # First check for direct keywords
    continuation_keywords = {
        "Content Agent": ["edit", "revise", "section", "add to", "update the"],
        "Finance Agent": ["compare", "analyze", "trend", "growth", "fundamentals"],
        "Web Search Agent": ["find", "search", "look up", "current", "latest"]
    }
    
    if any(keyword in user_input.lower() for keyword in continuation_keywords[agent_name]):
        return True
    
    # Then use LLM for contextual analysis
    try:
        continuity_agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            system_message=topic_continuity_prompt,
            llm_config={
                "config_list": [{
                    "model": "llama-3.3-70b-versatile",
                    "api_key": GROQ_API_KEY,
                    "temperature": 0,
                    "max_tokens": 1,
                }],
            },
        )
        
        response = rate_limiter.execute_with_retry(
            continuity_agent.run,
            topic_continuity_prompt.format(
                agent_name=agent_name,
                previous_context=truncate_text(previous_context, 1500),  # Optimized token usage
                user_input=truncate_text(user_input, 500)
            )
        )
        
        return "continue" in response.content.lower()
    except Exception as e:
        logging.error(f"Topic continuity check failed: {str(e)}")
        return False

# Enhanced Routing Prompt
routing_prompt = """Analyze the user input and context to select the best agent. Follow these rules:

1. Content Agent (Writing Tasks):
   - Keywords: write, edit, blog, section, content, draft, revise
   - Context: When referencing existing content ({context_last_response})
   - Examples: "Add a conclusion", "Improve the introduction"

2. Finance Agent (Financial Analysis):
   - Keywords: stock, compare, financial, EPS, revenue, dividend
   - Context: When mentioning tickers or financial metrics
   - Examples: "Compare AAPL and MSFT", "Show dividend history"

3. Web Search Agent (General/Real-time):
   - Keywords: search, find, current, latest, how to, what is
   - Context: When needing fresh information ({context_last_search})
   - Default for ambiguous requests

Response Format:
<agent_name>

Current Context:
- Previous Response: {context_last_response}
- Previous Search: {context_last_search}
- Previous Finance: {context_last_finance}

Conversation History:
{history}

User Input: "{input}"
Selected Agent: """

try:
    routing_agent = Agent(
        name="Routing Agent",
        system_message=routing_prompt,
        model=Groq(id="llama-3.3-70b-versatile"),
        llm_config={
            "config_list": [autogen_config | {
                "model": "llama-3.3-70b-versatile",
                "api_key": GROQ_API_KEY,
                "api_type": "groq",
                "temperature": 0,
                "max_tokens": 15,
            }],
        },
    )
    logging.info("Agents initialized successfully")
except Exception as e:
    logging.error(f"Agent initialization failed: {str(e)}", exc_info=True)
    raise

# Define Agents
try:
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for the information",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tools_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance AI Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            ),
        ],
        instructions=["Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    # LLM Configuration
    config_list = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": GROQ_API_KEY,
        "api_type": "groq",
    }]
    llm_config = {"config_list": config_list}

    # Create Agents
    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="""You are a professional writer specializing in creating engaging, well-structured, and SEO-friendly blog posts. Your role is to:
        1. Write high-quality content based on the given topic or request.
        2. Ensure the content includes:
        - A compelling title optimized for search engines
        - Clear and concise headings (H1, H2, H3)
        - Proper use of keywords without overstuffing
        - Engaging introduction and conclusion
        - Well-structured paragraphs with smooth transitions
        3. Incorporate feedback from reviewers (SEO, Legal, Ethics, Critic) to refine the content.
        4. Avoid adding comments or explanations in your output—only return the final polished version of the content.""",
        llm_config=llm_config,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="""You are a content critic specializing in providing constructive feedback to improve writing quality. Your role is to:
        1. Review the content for:
        - Clarity and readability
        - Logical flow and structure
        - Grammar, spelling, and punctuation
        - Tone and style consistency
        - Engagement and audience appeal
        2. Provide actionable feedback to help the writer refine their work.
        3. Focus on both strengths and areas for improvement.""",
        llm_config=llm_config,
    )

    SEO_reviewer = autogen.AssistantAgent(
        name="SEO Reviewer",
        system_message="""You are an SEO expert specializing in content optimization for search engines. Your role is to:
        1. Analyze the content for SEO best practices, including:
        - Keyword usage and placement (title, headings, meta description, body)
        - Content structure and readability (headings, paragraph length, bullet points)
        - Internal and external linking opportunities
        - Meta tags and alt text for images
        - Mobile-friendliness and page load speed considerations
        2. Identify opportunities to improve search engine rankings, such as:
        - Adding relevant keywords or long-tail phrases
        - Improving content depth and relevance to target queries
        - Enhancing user engagement signals (e.g., clear CTAs, interactive elements)
        3. Provide actionable suggestions to improve SEO performance.""",
        llm_config=llm_config,
    )

    legal_reviewer = autogen.AssistantAgent(
        name="Legal Reviewer",
        system_message="""You are a legal reviewer specializing in content compliance and risk mitigation. Your role is to:
        1. Identify potential legal issues in the content, including:
        - Defamation or libel risks
        - Intellectual property violations (copyright, trademarks)
        - Privacy law compliance (GDPR, CCPA, etc.)
        - Regulatory compliance (industry-specific rules)
        - Contractual obligations or disclosures
        2. Highlight areas that may require disclaimers or disclosures.
        3. Flag any statements that could be considered misleading or deceptive.
        4. Ensure proper attribution of sources and references.
        5. Suggest modifications to mitigate legal risks.""",
        llm_config=llm_config,
    )

    ethics_reviewer = autogen.AssistantAgent(
        name="Ethics Reviewer",
        system_message="""You are an ethics reviewer specializing in ensuring content is ethically sound and free from issues. Your role is to:
        1. Identify potential ethical concerns in the content, including:
        - Bias, discrimination, or stereotyping
        - Misrepresentation of facts or data
        - Harmful or misleading information
        - Privacy violations or unethical data usage
        - Conflicts of interest or lack of transparency
        2. Evaluate the tone and language for inclusivity, fairness, and respect.
        3. Ensure the content aligns with ethical standards and best practices.""",
        llm_config=llm_config,
    )

    meta_reviewer = autogen.AssistantAgent(
        name="Meta Reviewer",
        system_message="""You are a meta reviewer responsible for aggregating and synthesizing feedback from other reviewers (SEO, Legal, Ethics). Your role is to:
        1. Review all feedback provided by other reviewers.
        2. Identify patterns or recurring issues in the feedback.
        3. Prioritize the most critical changes needed to improve the content.
        4. Provide a final, consolidated set of recommendations.""",
        llm_config=llm_config,
    )

    # Reflection Message Function
    def reflection_message(recipient, messages, sender, config):
        return f'''Review the following content: 
        \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

    # Define Review Chats
    review_chats = [
        {
            "recipient": SEO_reviewer,
            "message": reflection_message,
            "summary_method": "reflection_with_llm",
            "summary_args": {"summary_prompt": 
                            "Return review as JSON only: {'Reviewer': '', 'Review': ''}."},
            "max_turns": 1,
        },
        {
            "recipient": legal_reviewer,
            "message": reflection_message,
            "summary_method": "reflection_with_llm",
            "summary_args": {"summary_prompt": 
                            "Return review as JSON only: {'Reviewer': '', 'Review': ''}."},
            "max_turns": 1,
        },
        {
            "recipient": ethics_reviewer,
            "message": reflection_message,
            "summary_method": "reflection_with_llm",
            "summary_args": {"summary_prompt": 
                            "Return review as JSON only: {'reviewer': '', 'review': ''}."},
            "max_turns": 1,
        },
        {
            "recipient": meta_reviewer,
            "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
            "max_turns": 1,
        },
    ]

    # Nested Review Registration
    critic.register_nested_chats(
        review_chats,
        trigger=writer,
    )
    logging.info("All agents initialized successfully")
except Exception as e:
    logging.error(f"Agent setup failed: {str(e)}", exc_info=True)
    raise

def format_history(chat_history: list) -> str:
    return "\n".join([f"User: {msg[0]}\nAssistant: {msg[1]}" for msg in chat_history])

def process_message(user_input: str, history: list, state: dict) -> Iterator[tuple[list, dict]]:
    # Initialize new_state with a copy of current state at the beginning
    new_state = state.copy()
    
    try:
        logging.info(f"Processing: {user_input[:100]}...")
        formatted_history = format_history(history)
        response_content = ""
        agent_name = "System"
        selected_agent = "Web Search Agent"

        # Get last response context with proper null handling
        last_response = state.get('last_response', '')
        last_agent = state.get('last_agent', 'None')

        try:
            # Get routing decision with proper null handling
            routing_query = routing_prompt.format(
                input=user_input,
                context_last_response=truncate_text(last_response, 500) if last_response else 'None',
                context_last_search=truncate_text(state.get('last_search', ''), 500) if state.get('last_search') else 'None',
                context_last_finance=truncate_text(state.get('last_finance', ''), 500) if state.get('last_finance') else 'None',
                history=truncate_text(formatted_history, 1000)  # Limit history tokens
            )
            
            routing_response: RunResponse = rate_limiter.execute_with_retry(
                routing_agent.run,
                routing_query
            )
            selected_agent = routing_response.content.strip().replace('.', '').strip().title()
            valid_agents = ["Content Agent", "Finance Agent", "Web Search Agent"]
            selected_agent = selected_agent if selected_agent in valid_agents else "Web Search Agent"
            logging.info(f"Routing to: {selected_agent}")

            # Topic continuity check
            is_continuation = check_topic_continuity(
                user_input=user_input,
                previous_context=last_response,  # Use last_response as previous_context
                agent_name=last_agent
            )

            # Build agent context
            context = f"Conversation History:\n{truncate_text(formatted_history, 2000)}\n"
            if is_continuation and last_response:
                context += f"Previous Context ({last_agent}):\n{truncate_text(last_response, 2000)}\n"
            context += f"Current Request: {user_input}"

            # Agent Execution
            if selected_agent == "Finance Agent":
                response = rate_limiter.execute_with_retry(
                    finance_agent.run,
                    context
                )
                response_content = response.content if response else "Failed to get financial data."
                agent_name = "Finance Agent"
                new_state['last_finance'] = response_content

            elif selected_agent == "Content Agent":
                try:
                    # Build content prompt with available contexts
                    content_prompt = f"Create content based on:\n"
                    if state.get('last_search'):
                        content_prompt += f"Search Results:\n{truncate_text(state['last_search'], 2000)}\n"
                    if state.get('last_finance'):
                        content_prompt += f"Financial Data:\n{truncate_text(state['last_finance'], 2000)}\n"
                    content_prompt += f"User Request: {user_input}"

                    res = rate_limiter.execute_with_retry(
                        critic.initiate_chat,
                        recipient=writer,
                        message=content_prompt,
                        max_turns=2
                    )
                    response_content = res.summary if res else "Content creation failed."
                    agent_name = "Content Agent"
                    new_state['last_content'] = response_content

                except Exception as e:
                    response_content = f"Content error: {str(e)}"
                    logging.error(f"Content agent error: {str(e)}")

            else:  # Web Search Agent
                response = rate_limiter.execute_with_retry(
                    web_search_agent.run,
                    context
                )
                response_content = response.content if response else "Web search failed."
                agent_name = "Web Search Agent"
                new_state['last_search'] = response_content

            # Update global context tracking
            new_state['last_response'] = response_content
            new_state['last_agent'] = selected_agent

        except Exception as e:
            response_content = f"System error: {str(e)}"
            logging.critical(f"Processing failure: {traceback.format_exc()}")

        # Stream response
        partial_response = ""
        try:
            if not response_content:
                response_content = "No response generated."
            for character in response_content:
                partial_response += character
                time.sleep(0.02)
                yield history + [(user_input, partial_response)], new_state
        except Exception as e:
            logging.error(f"Streaming interrupted: {str(e)}")
            yield history + [(user_input, "Response streaming failed")], new_state

        # Update state and save to DB
        full_response = history + [(user_input, response_content)]
        try:
            db_session = SessionLocal()
            chat_record = ChatMessage(
                user_input=user_input[:500],
                agent_name=agent_name,
                agent_response=response_content[:10000],
                timestamp=datetime.utcnow()
            )
            db_session.add(chat_record)
            db_session.commit()
        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")
        finally:
            db_session.close()

        yield full_response, new_state

    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logging.critical(f"Unhandled exception: {traceback.format_exc()}")
        yield history + [(user_input, error_msg)], new_state


# Function to load chat history from the database
def load_chat_history():
    try:
        db_session = SessionLocal()
        chat_history = db_session.query(ChatMessage).order_by(ChatMessage.timestamp.asc()).all()
        formatted_history = []
        for message in chat_history:
            formatted_history.append([message.user_input, message.agent_response])
        db_session.close()
        return formatted_history
    except Exception as e:
        logging.error(f"Failed to load chat history: {str(e)}")
        return []

# Function to clear chat history from the database
def clear_chat_history():
    try:
        db_session = SessionLocal()
        db_session.query(ChatMessage).delete()
        db_session.commit()
        db_session.close()
        logging.info("Chat history cleared from the database")
        return []
    except Exception as e:
        logging.error(f"Failed to clear chat history: {str(e)}")
        return []

# Gradio Interface with State Management
with gr.Blocks(title="AI Agent Collaboration Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Agent Collaboration Hub")
    gr.Markdown("Interact with specialized AI agents with context-aware responses")
    
    chatbot = gr.Chatbot(height=500, label="Conversation")
    msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
    clear = gr.Button("Clear History")
    state = gr.State({
        'last_content': None,
        'last_search': None,
        'last_finance': None,
        'last_response': None,
        'last_agent': None
    })

    # Load initial chat history when the page loads
    def load_initial_history():
        return load_chat_history()
    
    demo.load(
        load_initial_history,
        outputs=[chatbot]
    )

    def user(user_message, history, state):
        logging.info(f"User message received: {user_message}")
        return "", history + [[user_message, None]], state

    def bot(history, state):
        if not history or not history[-1][0]:
            return history + [("", "How can I help you today?")], state
        
        user_input = history[-1][0]
        try:
            response_gen = process_message(user_input, history[:-1], state)
            for response in response_gen:
                updated_history, updated_state = response
                if len(updated_history) == len(history) + 1:
                    history, state = updated_history, updated_state
                else:
                    history[-1] = [user_input, updated_history[-1][1]]
                yield history, state
        except Exception as e:
            error_msg = f"System Error: {str(e)}"
            logging.error(f"Bot error: {traceback.format_exc()}")
            yield history + [("", error_msg)], state

    msg.submit(user, [msg, chatbot, state], [msg, chatbot, state], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )
    
    # Updated clear button with proper DB cleanup and history reset
    clear.click(
        fn=lambda: (None, {'last_content': None, 'last_search': None, 'last_finance': None}),
        outputs=[chatbot, state],
        queue=False
    ).then(
        clear_chat_history,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    logging.info("Launching Gradio interface")
    try:
        demo.launch()
    except Exception as e:
        logging.critical(f"Application failed to start: {str(e)}", exc_info=True)
        raise