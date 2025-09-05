import os
import json
from azure.identity import DefaultAzureCredential
from azure.ai.language.conversations import ConversationAnalysisClient

class CLUUtility:
    def __init__(self, config_path):
        self.load_config(config_path)
        os.environ['AZURE_CLIENT_ID'] = self.config['azure']['client_id']
        os.environ['AZURE_TENANT_ID'] = self.config['azure']['tenant_id']
        os.environ['AZURE_CLIENT_SECRET'] = self.config['azure']['client_secret']
        self.client = ConversationAnalysisClient(self.config['azure']['endpoint'], DefaultAzureCredential())

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def load_training_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def train_model(self):
        data = self.load_training_data('data/training_data.json')
        project = self.config['project']
        training = self.config['training']
        
        print("Training the model...")
        self.client.train(
            project_name=project['project_name'],
            deployment_name=project['deployment_name'],
            training_mode=training['training_mode'],
            data_splitting=training['data_splitting']
        )
        print("Model training completed.")

    def deploy_model(self):
        project = self.config['project']
        
        print("Deploying the model...")
        self.client.deploy(
            project_name=project['project_name'],
            deployment_name=project['deployment_name'],
            model_name=project['model_name']
        )
        print("Model deployed.")

    def analyze_query(self, query):
        project = self.config['project']
        response = self.client.analyze(
            project_name=project['project_name'],
            deployment_name=project['deployment_name'],
            query=query
        )
        return response

    def validate(self, queries):
        for query in queries:
            response = self.analyze_query(query)
            print(response)
