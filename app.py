from flask import Flask, request, jsonify
from transformers import pipeline
import re
import string

# Initialize Flask app
app = Flask(__name__)

# Initialize the text-classification pipeline
pipe = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(f"[{string.punctuation}]", "", text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Additional preprocessing steps (e.g., stopword removal) can be added here

    return text

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data (text) from the POST request
    data = request.get_json()

    # Check if text is in the input data
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']

    # Preprocess the input text
    cleaned_text = preprocess_text(text)

    # Use the pipeline to classify the preprocessed text
    result = pipe(cleaned_text)

    # Return the prediction result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
