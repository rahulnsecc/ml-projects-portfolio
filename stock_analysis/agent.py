import autogen
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
load_dotenv()
# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your Groq API key in environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")   # Get free API key from newsapi.org

# Initialize News API
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Configure Groq
# LLM Configuration
config_list = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": GROQ_API_KEY,  # Replace with your API key
    "api_type": "groq",
}]
llm_config = {"config_list": config_list}


# Agent Definitions
def _is_termination_msg(content):
    return content.get("content", "").rstrip().endswith("TERMINATE")
'''legal_reviewer = autogen.AssistantAgent(
    name="Legal Reviewer",
    system_message="You are a legal reviewer, ensuring content is legally compliant and free from potential issues. "
                   "Provide suggestions in concise, actionable bullet points.",
    llm_config=llm_config,
)'''
# Data Collector Agent
data_collector = autogen.AssistantAgent(
    name="Data_Collector",
    system_message="You collect and organize financial data. Reply TERMINATE when done.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False
)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def get_news(ticker):
    news = newsapi.get_everything(q=ticker,
                                 language='en',
                                 sort_by='relevancy',
                                 page_size=5)
    return [article['title'] + ": " + article['description'] for article in news['articles']]

data_collector.register_function(
    function_map={
        "get_stock_data": get_stock_data,
        "get_news": get_news
    }
)

# Technical Analysis Agent
technical_analyst = autogen.AssistantAgent(
    name="Technical_Analyst",
    system_message="You perform technical analysis. Calculate indicators and identify trends.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False
)

def calculate_technical_indicators(data):
    return {
        "RSI": ta.rsi(data['Close'], length=14).iloc[-1],
        "MACD": ta.macd(data['Close']).iloc[-1].to_dict(),
        "Bollinger_Bands": ta.bbands(data['Close']).iloc[-1].to_dict(),
        "SMA_50": ta.sma(data['Close'], 50).iloc[-1],
        "SMA_200": ta.sma(data['Close'], 200).iloc[-1]
    }

technical_analyst.register_function(
    function_map={"calculate_technical_indicators": calculate_technical_indicators}
)

# Fundamental Analysis Agent
fundamental_analyst = autogen.AssistantAgent(
    name="Fundamental_Analyst",
    system_message="Analyze financial statements and valuation metrics.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False
)

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    return {
        "financials": stock.financials.iloc[:, :3].to_dict(),
        "balance_sheet": stock.balance_sheet.iloc[:, :3].to_dict(),
        "cash_flow": stock.cashflow.iloc[:, :3].to_dict(),
        "valuation": stock.info.get('forwardPE')
    }

fundamental_analyst.register_function(
    function_map={"get_fundamentals": get_fundamentals}
)

# Sentiment Analysis Agent
sentiment_analyst = autogen.AssistantAgent(
    name="Sentiment_Analyst",
    system_message="Analyze news sentiment and market mood.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Machine Learning Agent
ml_engineer = autogen.AssistantAgent(
    name="ML_Engineer",
    system_message="Predict future prices using machine learning models.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False
)

def predict_prices(data):
    data = data.reset_index()
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, predictions),
        "last_prediction": predictions[-1],
        "actual_last_price": target.iloc[-1]
    }

ml_engineer.register_function(function_map={"predict_prices": predict_prices})

# Chief Investment Officer Agent
cio = autogen.AssistantAgent(
    name="Chief_Analyst",
    system_message="""Synthesize all analyses into investment recommendations. Consider:
    - Technical indicators
    - Fundamental valuation
    - Market sentiment
    - ML predictions
    - Macroeconomic factors
    Provide detailed reasoning. End with 'TERMINATE'""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# User Proxy
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    default_auto_reply="Continue",
    is_termination_msg=_is_termination_msg
)

# Agent Workflow
def analyze_stock(ticker):
    groupchat = autogen.GroupChat(
        agents=[user_proxy, data_collector, technical_analyst, 
                fundamental_analyst, sentiment_analyst, ml_engineer, cio],
        messages=[],
        max_round=20
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    user_proxy.initiate_chat(
        manager,
        message=f"""Analyze {ticker} stock comprehensively. Consider:
        1. Historical price data
        2. Technical indicators
        3. Financial fundamentals
        4. News sentiment
        5. ML predictions
        Provide final recommendation with detailed reasoning."""
    )

# Run Analysis
if __name__ == "__main__":
    analyze_stock("AAPL")