import numpy as np
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

def get_hashing_features(texts):
    # Initialize HashingVectorizer object with desired parameters
    vectorizer = HashingVectorizer(analyzer='word', token_pattern=r'\b\w+\b', n_features=10000)

    # Transform texts into feature vectors
    features = vectorizer.transform(texts)
    
    joblib.dump(vectorizer, 'models/vectorizer.pkl') # With vectorizer we don't need to use a word_to_index method

    # Convert feature vectors to numpy array for easier manipulation
    features = np.array(features.toarray()).astype(np.float32)

    return features
