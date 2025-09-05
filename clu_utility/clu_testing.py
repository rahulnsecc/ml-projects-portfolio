# Define the configuration for the training job
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
# Set up your Azure endpoint and key
deployment_name = "TransactionReportDeployment"
endpoint = ""
key=""
project_name = "TransactionReportBot"


# Initialize the analysis client
analysis_client = ConversationAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# Define the task for analysis
task = {
    "analysisInput": {
        "conversationItem": {
            "id": "1",
            "participantId": "1",
            "language": "en",
            "modality": "text",
            "text": "I need a transaction report for store number 123 from June 1st to June 10th.",
            "role": "user"
        }
    },
    "kind": "Conversation",
    "parameters": {
        "deploymentName": deployment_name,
        "projectName": project_name,
        "verbose": True,
        "stringIndexType": "Utf16CodeUnit"
    }
}

# Perform the analysis
response = analysis_client.analyze_conversation(
    task=task,
    content_type="application/json"
)

# Output the analysis results
print("Analysis completed successfully")
print(f"Query: {response['result']['query']}")
print(f"Top Intent: {response['result']['prediction']['topIntent']}")
print(f"Entities: {response['result']['prediction']['entities']}")