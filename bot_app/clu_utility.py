import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
from config import DefaultConfig

class CLUUtility:
    def __init__(self):
        self.client = ConversationAnalysisClient(
            endpoint=DefaultConfig.CLU_ENDPOINT,
            credential=AzureKeyCredential(DefaultConfig.CLU_API_KEY)
        )
        self.project_name = DefaultConfig.CLU_PROJECT_NAME
        self.deployment_name = DefaultConfig.CLU_DEPLOYMENT_NAME

    def analyze_input(self, text):
        task = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "participantId": "1",
                    "modality": "text",
                    "language": "en",
                    "text": text,
                    "role": "user"
                }
            },
            "parameters": {
                "projectName": self.project_name,
                "deploymentName": self.deployment_name,
                "verbose": True,
                "stringIndexType": "Utf16CodeUnit"
            }
        }
        try:
            response = self.client.analyze_conversation(
                task=task,
                content_type="application/json"
            )
            result = response["result"]["prediction"]
            print(result["topIntent"])
            return {
                "topIntent": result["topIntent"],
                "entities": result["entities"]
            }
        except Exception as e:
            print(f"Error analyzing input: {e}")
            return {"topIntent": "", "entities": {}}

# Save this file as clu_utility.py
