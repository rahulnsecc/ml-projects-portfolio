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


# Enterprise Chatbot with Azure Bot Framework

## 📌 Project Overview
Designed an **enterprise chatbot** using **Azure Bot Framework** that handles **50K+ queries/month**.  

---

## 🔑 Key Features
- **Azure AI Integration** (CLU, QnA Maker)  
- **Declarative Dialogue Management**  
- **Scalable Architecture**  
- **Automated Deployment**  

---

## 🛠️ Tech Stack
Azure Bot Framework, CLU, Python  

---

## 📂 Files
- `sop_bot.py`  
- `clu_creation.py`  


# Intelligent Batch Operations Monitoring

## 📌 Project Overview
Built a **predictive monitoring system** for batch workflows.  

---

## 🔑 Key Features
- **Pattern Analysis**  
- **Prognostic Monitoring**  
- **Stateful Alerting**  
- **Configurable System**  

---

## 🛠️ Tech Stack
Python, Pandas, ConfigParser, Logging  

---

## 📂 Files
- `analyze_file_patterns.py`  
- `check_file_arrivals.py`  

# Email Classification and Automation

## 📌 Project Overview
Built an **ML-powered email classifier** for **automated support ticket routing**.  

Uses **BERT embeddings + Logistic Regression**.

---

## 🔑 Key Features
- **NLP Embeddings** (BERT)  
- **ML Workflow** (MLflow)  
- **Automated Triage**  

---

## 🛠️ Tech Stack
Python, Scikit-learn, BERT, MLflow  

---

## 📂 Files
- `model.py`  
- `data/issue_data.csv`  

# Azure Cost Optimizer

## 📌 Project Overview
Developed a **cost optimization tool** for Azure resources.  

---

## 🔑 Key Features
- **Cost Analysis** (Azure SDK)  
- **Optimization Reports**  
- **Dashboard-Ready Outputs**  

---

## 🛠️ Tech Stack
Python, Azure SDK, Reporting  

---

## 📂 Files
- `app.py`  
- `config/`  

# Alert Monitoring App

## 📌 Project Overview
Developed an **ML-based monitoring system** for proactive incident detection.  

---

## 🔑 Key Features
- **Job Forecasting** (Prophet models)  
- **Threshold Alerts**  
- **Certificate Monitoring**  
- **Visualization Dashboard** (Flask + Chart.js)  

---

## 🛠️ Tech Stack
Python, Flask, Prophet, RandomForest, Chart.js  

---

## 📂 Files
- `app.py`  
- `src/alertmon/`  
- `src/scripts/train_models.py`  
- `static/` + `templates/`     
