import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train_cnn_model(features, labels, num_classes, test_size=0.5, epochs=10, batch_size=32): #Because of we have too much data I have used %50 of test size for prototype.
    """
    Trains a Convolutional Neural Network (CNN) model for sentiment analysis.

    Args:
        features (numpy.ndarray): Array of input features.
        labels (numpy.ndarray): Array of output labels.
        num_classes (int): Number of output classes.
        test_size (float): Size of the test set (default: 0.2).
        epochs (int): Number of training epochs (default: 10).
        batch_size (int): Batch size for training (default: 32).

    Returns:
        model (tensorflow.keras.models.Sequential): Trained CNN model.
        history (tensorflow.python.keras.callbacks.History): Training history object.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    # Reshape input features to match CNN input format
    input_shape = (features.shape[1], 1)
    X_train = X_train.reshape((X_train.shape[0], input_shape[0], input_shape[1]))
    X_test = X_test.reshape((X_test.shape[0], input_shape[0], input_shape[1]))

    # Initialize CNN model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model with desired loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model on training set
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model.save('./models/cnn_model.h5')
    
    return model, history
