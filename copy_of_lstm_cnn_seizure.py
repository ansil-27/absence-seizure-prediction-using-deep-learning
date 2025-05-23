# -*- coding: utf-8 -*-
"""Copy of LSTM_CNN_seizure.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QUgMEldm7EuXHZ0gx_2CKerTBk8s5lxU
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_data_from_folder(folder_path, label):
    data_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        data = np.loadtxt(file_path)
        data_list.append(data)
    data_array = np.vstack(data_list)
    labels = np.array([label] * data_array.shape[0])
    return data_array, labels


paths = {
    'A': '/content/drive/MyDrive/epilepsy seizure/dataset_split/Z_split', # Class 0: Healthy with eyes open
    'B': '/content/drive/MyDrive/epilepsy seizure/dataset_split/O_split', # Class 1: Healthy with eyes closed
    'C': '/content/drive/MyDrive/epilepsy seizure/dataset_split/N_split', # Class 2: Seizure-free intervals from the hippocampal formation
    'D': '/content/drive/MyDrive/epilepsy seizure/dataset_split/F_split', # Class 3: Seizure activity
    'E': '/content/drive/MyDrive/epilepsy seizure/dataset_split/S_split'  # Class 4: Seizure activity
}

def prepare_data_for_classification(classes):
    X, y = [], []
    for label, class_id in enumerate(classes):
        folder_path = paths[class_id]
        data, labels = load_data_from_folder(folder_path, label)
        X.append(data)
        y.append(labels)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

def create_cnn_lstm_model(input_shape, filters, kernel_size, pool_size, lstm_units1, lstm_units2):
    model = Sequential() # Creates a Sequential model, which is a linear stack of layers.

    # Convolutional layer
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape)) # Adds a 1D convolutional layer to the model for feature extraction.
    model.add(MaxPooling1D(pool_size=pool_size)) # Adds a max pooling layer to downsample the feature maps.

    # LSTM layers
    model.add(LSTM(units=lstm_units1, return_sequences=True)) # Adds the first LSTM layer with the specified number of units and returns sequences for the next LSTM layer.
    model.add(LSTM(units=lstm_units2)) # Adds the second LSTM layer with the specified number of units.

    # Dense layer for binary classification
    model.add(Dense(units=1, activation='sigmoid'))  # Adds a dense layer with a sigmoid activation for binary classification.

    # Compile the model with binary crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compiles the model with the Adam optimizer, binary crossentropy loss, and accuracy metric.

    return model # Returns the compiled CNN-LSTM model.

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    # Sensitivity (Recall for the positive class, assuming 1 is the positive label)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    # Specificity (Recall for the negative class, assuming 0 is the negative label)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    # F1 score
    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1

def train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test, filters, kernel_size, pool_size, lstm_units1, lstm_units2, batch_size, epochs):
    # Create the model
    model = create_cnn_lstm_model((X_train.shape[1], 1), filters, kernel_size, pool_size, lstm_units1, lstm_units2)

    # Display the model architecture
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred_labels = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

    # Compute additional metrics (sensitivity, specificity, F1 score)
    accuracy, sensitivity, specificity, f1 = compute_metrics(y_test, y_pred_labels)
    print(f'Accuracy: {accuracy}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'F1 Score: {f1}')

    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()

    # Confusion matrix as heatmap
    conf_matrix = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    model.save('/content/drive/MyDrive/epilepsy seizure/my_cnn_lstm_model_epilepsyornot.h5')

    return history, score

# Load and preprocess data for classes A-B vs. C-D-E
classes_ab = ['A', 'B']
classes_cde = ['C', 'D']
X_ab, y_ab = prepare_data_for_classification(classes_ab)
X_cde, y_cde = prepare_data_for_classification(classes_cde)

# Merge and create new binary labels
X = np.concatenate([X_ab, X_cde])
y = np.concatenate([np.zeros(len(y_ab)), np.ones(len(y_cde))])  # 0 for A-B, 1 for C-D-E

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for RNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN-LSTM model
train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test, filters=64, kernel_size=3, pool_size=2, lstm_units1=128, lstm_units2=128, batch_size=32, epochs=30)

# Load and preprocess data for classes A and E
classes_a = ['A']
classes_e = ['E']
X_a, y_a = prepare_data_for_classification(classes_a)
X_e, y_e = prepare_data_for_classification(classes_e)

# Merge and create new labels
X = np.concatenate([X_a, X_e])
y = np.concatenate([np.zeros(len(y_a)), np.ones(len(y_e))])  # 0 for A, 1 for E

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for RNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN-LSTM model
train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test, filters=64, kernel_size=3, pool_size=2, lstm_units1=128, lstm_units2=128, batch_size=32, epochs=5)

# Load and preprocess data for classes B and E
classes_b = ['A','B']
classes_e = ['E']
X_b, y_b = prepare_data_for_classification(classes_b)
X_e, y_e = prepare_data_for_classification(classes_e)

# Merge and create new labels
X = np.concatenate([X_b, X_e])
y = np.concatenate([np.zeros(len(y_b)), np.ones(len(y_e))])  # 0 for B, 1 for E

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for RNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN-LSTM model
train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test, filters=64, kernel_size=3, pool_size=2, lstm_units1=128, lstm_units2=128, batch_size=32, epochs=5)

# Load and preprocess data for classes B and D
classes_bd = ['A', 'C']
X, y = prepare_data_for_classification(classes_bd)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for CNN-LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN-LSTM model
train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test,
                                  filters=128, kernel_size=5,
                                  pool_size=2,
                                  lstm_units1=256, lstm_units2=128,
                                  batch_size=32,
                                  epochs=10)

# Load and preprocess data for classes B and D
classes_bd = ['B', 'D']
X, y = prepare_data_for_classification(classes_bd)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for CNN-LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN-LSTM model
train_and_evaluate_cnn_lstm_model(X_train, y_train, X_test, y_test,
                                  filters=128, kernel_size=5,
                                  pool_size=2,
                                  lstm_units1=256, lstm_units2=128,
                                  batch_size=32,
                                  epochs=20)

