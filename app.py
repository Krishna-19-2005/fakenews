from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Flask app
app = Flask(__name__)

# Load and preprocess the dataset
dataset = pd.read_csv(r"C:\Users\klpna\Downloads\archive\news_dataset.csv")
en_data = dataset[["label"]]
ohe = OneHotEncoder(drop="first")
ar = ohe.fit_transform(en_data).toarray()
dataset["label"] = pd.DataFrame(ar, columns=[['label_FAKE']])

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)  # Join tokens back into string

# Apply preprocessing
dataset["text"] = dataset["text"].astype(str)
dataset["text"] = dataset["text"].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset["text"])
y = dataset['label']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def make_prediction(model, vectorizer, text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Define routes for Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_input = request.form['news_input']
    prediction = make_prediction(model, vectorizer, news_input)
    return jsonify({"prediction": "Fake" if prediction == 0 else "Real"})

if __name__ == '__main__':
    app.run(debug=True)
