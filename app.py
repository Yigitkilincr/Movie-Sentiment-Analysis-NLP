import pandas as pd
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from src.preprocessing import preprocess_data
from src.feature_extraction import get_bag_of_words_features
from src.model_training import train_cnn_model
from src.model_evaluation import evaluate_model

app = Flask(__name__)
'''
##Preprocess data
raw_data = pd.read_csv('data/raw/IMDB Dataset.csv')
print(raw_data.head())
print(raw_data['sentiment'].value_counts())

preprocessed_reviews = raw_data['review'].apply(preprocess_data)
preprocessed_data = pd.DataFrame({'review': preprocessed_reviews, 'sentiment': raw_data['sentiment']})

if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
preprocessed_data.to_csv('data/processed/IMDB Processed Data.csv', index=False)

##Preprocessing closed until bugs in the code are fixed
'''

##Train CNN model
preprocessed_data = pd.read_csv('data/processed/IMDB Processed Data.csv')

labels = pd.get_dummies(preprocessed_data['sentiment']).values
features = get_bag_of_words_features(preprocessed_data['review'], max_words=10000)

model, history = train_cnn_model(features, labels, num_classes=2)

# Save trained model and word-to-index mapping
model.save('models/cnn_model.h5')
np.save('models/word_to_index.npy', get_bag_of_words_features.word_to_index)


model = tf.keras.models.load_model('models/cnn_model.h5')
word_to_index = np.load('models/word_to_index.npy', allow_pickle=True).item()

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from request body
    text = request.json['text']

    # Preprocess input text
    preprocessed_text = preprocess_data(text)

    # Create bag-of-words feature vector
    features = get_bag_of_words_features([preprocessed_text], word_to_index)

    # Make prediction using trained model
    prediction = model.predict(features)
    prediction = np.argmax(prediction, axis=1)

    # Return prediction result as JSON object
    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Load test set and labels
    test_features = np.load('data/processed/test_features.npy')
    test_labels = np.load('data/processed/test_labels.npy')

    # Evaluate performance of trained model on test set
    metrics = evaluate_model(model, test_features, test_labels)

    # Return evaluation metrics as JSON object
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)

