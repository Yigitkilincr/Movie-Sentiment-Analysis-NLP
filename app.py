import pandas as pd
import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, jsonify, request, render_template
from src.preprocessing import preprocess_data
from src.feature_extraction import get_hashing_features
from src.model_training import train_cnn_model
from src.model_evaluation import evaluate_model

app = Flask(__name__)
'''

!!! Note !!! : You must run the commented part if you are running the app first time.

##Preprocess data
raw_data = pd.read_csv('data/raw/IMDB Dataset.csv')
print(raw_data.head())
print(raw_data['sentiment'].value_counts())

preprocessed_reviews = raw_data['review'].apply(preprocess_data)
preprocessed_data = pd.DataFrame({'review': preprocessed_reviews, 'sentiment': raw_data['sentiment']})

if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
preprocessed_data.to_csv('data/processed/IMDB Processed Data.csv', index=False)

##Train CNN model
preprocessed_data = pd.read_csv('data/processed/IMDB Processed Data.csv') #Note : Do not forget to check your path that must be same path as data.

labels = pd.get_dummies(preprocessed_data['sentiment']).values
features = get_hashing_features(preprocessed_data['review'])


model, history = train_cnn_model(features, labels, num_classes=2)

# Save trained model and word-to-index mapping

model.save('models/cnn_model.h5')
'''

## Load pre-trained model and vectorizer
model = tf.keras.models.load_model('models/cnn_model.h5')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from request body
    text = request.form['text']

    # Preprocess input text
    preprocessed_text = preprocess_data(text)

    # Create bag-of-words feature vector
    features = get_hashing_features([preprocessed_text])

    # Make prediction using trained model
    prediction = model.predict(features)
    prediction = np.argmax(prediction, axis=1)

    # Return prediction result as JSON object
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify(sentiment=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
