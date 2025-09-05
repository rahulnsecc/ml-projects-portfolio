from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.core.credentials import AzureKeyCredential
from azure.core.polling import LROPoller
# Set up your Azure endpoint and key

endpoint = ""
key="="

# Initialize the authoring client
client = ConversationAuthoringClient(endpoint=endpoint, credential=AzureKeyCredential(key))
project_name = "TransactionReportBot"
# Define the configuration for the training job
training_configuration = {
    "modelLabel": "TransactionReportModel",
    "trainingMode": "standard",  # Use "advanced" if you require more complex training
    "evaluationOptions": {
        "kind": "percentage",
        "testingSplitPercentage": 20,  # 20% of data for testing
        "trainingSplitPercentage": 80  # 80% of data for training
    }
}

# Trigger the training job
train_response: LROPoller = client.begin_train(
    project_name=project_name,
    configuration=training_configuration,
    content_type="application/json"
)

# Wait for the training job to complete
train_result = train_response.result()

# Output the training result
print("Training completed successfully")
print(f"Model Label: {train_result['result']['modelLabel']}")
print(f"Training Status: {train_result['result']['trainingStatus']['status']}")
print(f"Evaluation Status: {train_result['result']['evaluationStatus']['status']}")
print(f"Training Start Time: {train_result['result']['trainingStatus']['startDateTime']}")
print(f"Training End Time: {train_result['result']['trainingStatus']['endDateTime']}")
print(f"Evaluation Start Time: {train_result['result']['evaluationStatus']['startDateTime']}")
print(f"Evaluation End Time: {train_result['result']['evaluationStatus']['endDateTime']}")
