import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient


# Set up your Azure endpoint and key
endpoint = ""
key=""

# Initialize the authoring client

client = ConversationAuthoringClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Define the project details
project_name = "TransactionReportBot"
project_details = {
    "metadata": {
        "language": "en",
        "projectKind": "Conversation",
        "projectName": project_name,
        "description": "A bot to handle transaction report requests",
        "multilingual": False,
        "settings": {
            "confidenceThreshold": 0.7
        }
    },
    "projectFileVersion": "2022-05-01",
    "stringIndexType": "Utf16CodeUnit",
    "assets": {
        "projectKind": "Conversation",
        "intents": [{"category": intent} for intent in ["RequestTransactionReport", "Confirm", "Cancel"]],
        "entities": [{"category": entity} for entity in ["startDate", "endDate", "storeNumber", "rxNumber"]],
        "utterances": []
    }
}

# Load training data from CSV
training_data = pd.read_csv('training_data.csv')

# Parse training data and add to project details
for index, row in training_data.iterrows():
    text = row['text']
    intent = row['intent']
    entity_data = row['entities']

    entities = []
    if isinstance(entity_data, str):
        # Parse entity data if it is a string
        entity_pairs = entity_data.split(';')
        for pair in entity_pairs:
            if ':' in pair:
                entity, value = pair.split(':')
                offset = text.index(value)
                length = len(value)
                entities.append({"category": entity, "offset": offset, "length": length})

    project_details["assets"]["utterances"].append({
        "intent": intent,
        "text": text,
        "entities": entities
    })

# Import the project
import_response = client.begin_import_project(
    project_name=project_name,
    project=project_details,
    exported_project_format="Conversation"
)

# Wait for the import operation to complete
import_response.result()

print("Project imported successfully")