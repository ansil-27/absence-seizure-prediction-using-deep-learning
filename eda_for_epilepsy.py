# -*- coding: utf-8 -*-
"""EDA_for_epilepsy.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oFk_hNsjjCuK668M4VB_UzDFPk8spuxt
"""

import os
import numpy as np

# Define paths to each class
paths = {
    'A': '/content/drive/MyDrive/epilepsy seizure/dataset_split/Z_split', # Class A: Healthy (No Seizure)
    'B': '/content/drive/MyDrive/epilepsy seizure/dataset_split/O_split', # Class B: Healthy (No Seizure)
    'C': '/content/drive/MyDrive/epilepsy seizure/dataset_split/N_split', # Class C: Preictal (Seizure-free)
    'D': '/content/drive/MyDrive/epilepsy seizure/dataset_split/F_split', # Class D: Preictal (Seizure-free)
    'E': '/content/drive/MyDrive/epilepsy seizure/dataset_split/S_split'  # Class E: Seizure activity
}

# Combined dataset and labels
data = []
labels = []

# Class mapping
class_mapping = {
    'A': 0,  # No Seizure
    'B': 0,  # No Seizure
    'C': 1,  # Potential Seizure
    'D': 1,  # Potential Seizure
    'E': None  # Excluded
}

# Load data from .txt files
for class_label, path in paths.items():
    if class_mapping[class_label] is None:  # Skip Class E
        continue

    label = class_mapping[class_label]
    for file in os.listdir(path):
        if file.endswith('.txt') or file.endswith('.TXT'):  # Ensure correct file format
            file_path = os.path.join(path, file)
            chunk_data = np.loadtxt(file_path)  # Load data points from .txt file
            data.append(chunk_data)
            labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Data shape: {data.shape}")  # (num_samples, 170)
print(f"Labels shape: {labels.shape}")  # (num_samples,)

# Print first few data samples and labels
print("Sample data (first chunk):", data[0])
print("Sample labels (first few):", labels[:5])

# Check class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class Distribution:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")

import matplotlib.pyplot as plt
import random
import numpy as np

# Define class names for better labeling
class_names = {
    'A': "Class A: Healthy (Eyes Open)",
    'B': "Class B: Healthy (Eyes Closed)",
    'C': "Class C: Preictal (Seizure-Free, Outside Zone)",
    'D': "Class D: Preictal (Seizure-Free, Within Zone)",
    'E': "Class E: Ictal (Seizure Activity)"
}

# Plot a few signals from each class
for class_label, path in paths.items():
    # Skip visualization if the folder does not exist
    if not os.path.exists(path):
        print(f"Folder not found for {class_label}. Skipping visualization.")
        continue

    # List all .txt files in the folder
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt') or f.endswith('.TXT')]

    # Select a random file
    if len(txt_files) > 0:
        sample_file = random.choice(txt_files)
        sample_path = os.path.join(path, sample_file)

        # Load the signal
        signal = np.loadtxt(sample_path)

        # Plot the signal
        plt.figure(figsize=(10, 4))
        plt.plot(signal)
        plt.title(f"{class_names[class_label]} - {sample_file}")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")
        plt.show()
    else:
        print(f"No .txt files found in {path} for {class_label}.")

import os
import matplotlib.pyplot as plt

# Count the number of files in each class
class_file_counts = {}

for class_label, path in paths.items():
    num_files = len([file for file in os.listdir(path) if file.endswith('.txt') or file.endswith('.TXT')])
    class_file_counts[class_label] = num_files

# Data for plotting
classes = list(class_file_counts.keys())
counts = list(class_file_counts.values())

# Plot the bar graph
plt.figure(figsize=(8, 6))
plt.bar(classes, counts, color=['blue', 'green', 'orange', 'red', 'purple'])
plt.xlabel('Classes')
plt.ylabel('Number of Files')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()