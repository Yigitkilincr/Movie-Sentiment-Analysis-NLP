import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_bag_of_words_features(texts):
    # Initialize CountVectorizer object with desired parameters
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b')

    # Fit vectorizer to texts and transform them into feature vectors
    features = vectorizer.fit_transform(texts)

    # Convert feature vectors to numpy array for easier manipulation
    features = np.array(features.toarray())

    # Get mapping of feature indices to feature names
    vocabulary = vectorizer.vocabulary_

    return features, vocabulary
