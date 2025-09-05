import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import datetime
from hyperopt import hp, fmin, tpe, Trials

# Define the search space for hyperparameter optimization
space = {
    "C": hp.loguniform("C", np.log(0.001), np.log(100)),
    "l1_ratio": hp.uniform("l1_ratio", 0, 1),
    "penalty": hp.choice("penalty", ["elasticnet"]),
    "solver": hp.choice("solver", ["saga"]),
    "random_state": hp.randint("random_state", 1000),
}

# Define objective function for hyperparameter optimization
def objective(params):
    # Extract hyperparameters
    C = params['C']
    l1_ratio = params['l1_ratio']
    penalty = 'elasticnet'
    solver = 'saga'
    random_state = params['random_state']

    # Train Logistic Regression classifier
    classifier = LogisticRegression(C=C, l1_ratio=l1_ratio, penalty=penalty, solver=solver, random_state=random_state, max_iter=1000)
    classifier.fit(X_train_bert, y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_test_bert)
    accuracy = accuracy_score(y_test, y_pred)

    return -accuracy  # Minimize negative accuracy (maximize accuracy)

# Read data
df = pd.read_csv('data/issue_data.csv')

# Encode labels
label_encoder = LabelEncoder()
df['issue_type_encoded'] = label_encoder.fit_transform(df['issue_category'])
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['issue'], df['issue_type_encoded'], test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode text data
def tokenize_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    return inputs

# Extract BERT embeddings
def get_bert_embeddings(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings.numpy()

# Convert text data to BERT embeddings
X_train_bert = np.vstack([get_bert_embeddings(tokenize_text(text)) for text in X_train])
X_test_bert = np.vstack([get_bert_embeddings(tokenize_text(text)) for text in X_test])

# Run hyperparameter optimization
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

print("Best parameters:", best_params)

# Train Logistic Regression classifier with best hyperparameters
best_classifier = LogisticRegression(
    C=best_params['C'],
    l1_ratio=best_params['l1_ratio'],
    penalty='elasticnet',
    solver='saga',
    random_state=best_params['random_state'],
    max_iter=1000
)
best_classifier.fit(X_train_bert, y_train)

# Evaluate classifier
y_pred = best_classifier.predict(X_test_bert)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with best hyperparameters:", accuracy)

# Log the model with MLflow
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(best_classifier, "model")

    # Fetch the best run ID
    best_run_id = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]['run_id']

    if best_run_id:
        # Load the best model
        best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

        # Print the path where to see the best model details
        print("Best model path:")
        print(os.path.abspath(f"runs:/{best_run_id}/model"))

        # Print the best model's parameters
        print("\nBest model parameters:")
        for param, value in best_model.get_params().items():
            print(f"{param}: {value}")

        # Define the source and destination paths
        source_path = f"mlruns/0/{best_run_id}/artifacts/model"
        destination_path = os.path.join(".", "model.pkl")

        # Rename existing model file if it exists
        if os.path.exists(destination_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = f"model.pkl_backup_{timestamp}"
            os.rename(destination_path, backup_path)
            print(f"Existing model file '{destination_path}' renamed to '{backup_path}'")

        # Move the model file to the destination folder
        os.rename(source_path, destination_path)

        print(f"Model moved from '{source_path}' to '{destination_path}'")
