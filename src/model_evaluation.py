import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, features, labels):
    """
    Evaluates the performance of a trained model on a test set.

    Args:
        model (tensorflow.keras.models.Sequential): Trained model to evaluate.
        features (numpy.ndarray): Array of input features.
        labels (numpy.ndarray): Array of output labels.

    Returns:
        metrics (dict): Dictionary of evaluation metrics (accuracy, precision, recall, F1 score).
    """
    # Reshape input features to match CNN input format
    input_shape = (features.shape[1], 1)
    features = features.reshape((features.shape[0], input_shape[0], input_shape[1]))

    # Predict output labels for test set
    y_pred = model.predict(features)
    y_pred = np.argmax(y_pred, axis=1)

    # Compute evaluation metrics
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred, average='weighted')
    recall = recall_score(labels, y_pred, average='weighted')
    f1 = f1_score(labels, y_pred, average='weighted')

    # Return dictionary of evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics
