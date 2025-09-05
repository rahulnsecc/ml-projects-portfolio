from flask import Flask, request, jsonify
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Load trained logistic regression classifier
classifier = joblib.load('model.pkl/model.pkl')

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

# Make predictions
def predict(text):
    # Tokenize and encode input text
    inputs = tokenize_text(text)
    bert_embeddings = get_bert_embeddings(inputs)

    # Make predictions
    predictions = classifier.predict(bert_embeddings)

    # Convert predictions to labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels.tolist()

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    text = data['text']

    predictions = predict(text)

    return jsonify({'predicted_category': predictions})

if __name__ == '__main__':
    app.run(debug=True)
