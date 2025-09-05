# CLU Utility – Azure Conversational Language Understanding

## 📌 Project Overview
The **CLU Utility** provides automation scripts for managing an **Azure Conversational Language Understanding (CLU)** project.  
It is part of the **Enterprise Chatbot** solution and streamlines training, deployment, and testing of CLU models used for intent and entity recognition.

---

## 🔑 Key Features
- **Automated Deployment** (`clu_deploy.py`)  
  Deploys trained CLU models to a target deployment, making them available for chatbot usage.  
  Includes model versioning, expiration details, and deployment metadata.

- **Model Testing** (`clu_testing.py`)  
  Provides an automated way to validate deployed models.  
  - Sends sample utterances (e.g., *“I need a transaction report for store number 123 from June 1st to June 10th”*).  
  - Extracts predicted **intents** and **entities** from the model response.  

- **Integration Ready**  
  Designed to be consumed by the **Enterprise Chatbot** (Azure Bot Framework) for intent-driven workflows.

---

## 🛠️ Tech Stack
- **Azure AI Language Services** (Conversation Authoring + Analysis)  
- **Python SDKs**:  
  - `azure.ai.language.conversations`  
  - `azure.core.credentials`  
- **Deployment + Testing Automation**  

---

## 📂 Files
- `clu_deploy.py` → Automates model deployment  
- `clu_testing.py` → Automates prediction testing  

---

## 🎯 Roles
- AI Product Engineer  
- LLM Engineer  
- GenAI & Data Systems Engineering Manager
