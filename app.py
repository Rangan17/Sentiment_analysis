
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return jsonify({'prediction': prediction})


