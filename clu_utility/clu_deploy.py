from azure.core.polling import LROPoller
import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient


# Set up your Azure endpoint and key
endpoint = ""
key=""

# Initialize the authoring client
client = ConversationAuthoringClient(endpoint=endpoint, credential=AzureKeyCredential(key))
project_name = "TransactionReportBot"
# Define deployment details
deployment_name = "TransactionReportDeployment"
deployment_details = {
    "trainedModelLabel": "TransactionReportModel"
}

# Deploy the project
deploy_response: LROPoller = client.begin_deploy_project(
    project_name=project_name,
    deployment_name=deployment_name,
    deployment=deployment_details,
    content_type="application/json"
)

# Wait for the deployment to complete
deployment_result = deploy_response.result()

# Output the deployment result
print("Deployment completed successfully")
print(f"Deployment Name: {deployment_result['deploymentName']}")
print(f"Model ID: {deployment_result['modelId']}")
print(f"Deployment Expiration Date: {deployment_result['deploymentExpirationDate']}")
print(f"Last Deployed DateTime: {deployment_result['lastDeployedDateTime']}")
