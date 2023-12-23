# %%
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
import random
import json
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

# %%
# Load the dataset from the Excel file (XLSX format)
CSV_PATH = "Training_Data.csv"  # Ganti dengan nama file CSV Anda

# Membaca data dari file CSV
df = pd.read_csv(CSV_PATH)

# %%
# Extract dataset features and labels from the DataFrame
X = df.drop(df.columns[-1], axis=1).values  # Fitur: Menghapus kolom terakhir
y = df[df.columns[-1]].values  # Label: Mengambil kolom terakhir

# %%
# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)  # Label = representasi biner

# %%
# Split the dataset into training and testing sets while preserving class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# %%
# Create a BPNN model with EarlyStopping, ada 3 hidden
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.summary()

# %%
# Compile the BPNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Train the BPNN model with EarlyStopping
start_time = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32, callbacks=[early_stopping])
end_time = time.time()
training_time = end_time - start_time

model.save("TrainingBPNN.h5")

# %%
# Define a mapping of the original labels to the desired labels
label_mapping = { 0: "Am", 1: "Bb", 2: "Bdim", 3: "C", 4: "Dm", 5: "Em", 6: "F", 7: "G"
}

# %%
# Evaluate the model and calculate classification results
test_loss, test_accuracy = model.evaluate(X_test, y_test)
train_loss, train_accuracy = model.evaluate(X_train, y_train)  # Added training accuracy

# Calculate predictions after training
predictions = model.predict(X_test)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))

# %%
# Evaluate the model and calculate classification results manually
y_test_argmax = np.argmax(y_test, axis=1)
y_pred_argmax = np.argmax(predictions, axis=1)

correct_predictions = np.sum(y_test_argmax == y_pred_argmax)
total_samples = len(y_test_argmax)
test_accuracy_manual = correct_predictions / total_samples

# Calculate confusion matrix manually
confusion_mat_manual = np.zeros((len(label_mapping), len(label_mapping)))

for i in range(total_samples):
    true_label = y_test_argmax[i]
    predicted_label = y_pred_argmax[i]
    confusion_mat_manual[true_label][predicted_label] += 1

# Convert confusion matrix to DataFrame for visualization
confusion_mat_df = pd.DataFrame(confusion_mat_manual, columns=label_mapping.values(), index=label_mapping.values())

# Output accuracy and confusion matrix
print(f"Testing Accuracy (Manual): {test_accuracy_manual}")
print("Confusion Matrix (Manual):")
print(confusion_mat_df)

# %%
# Save the classification results to a dictionary
classification_results = {
    "Testing Accuracy": test_accuracy,
    "Training Accuracy": train_accuracy,
    "Epoch": 300,
    "Training Time (seconds)": training_time,
    "Precision": classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), output_dict=True)
}

# %%
# Relabel the predicted and true labels
predicted_labels_renamed = [label_mapping.get(label, "Unknown") for label in np.argmax(predictions, axis=1)]
y_test_renamed = [label_mapping[label] for label in np.argmax(y_test, axis=1)]

# Recalculate the classification report and confusion matrix
classification_rep = classification_report(y_test_renamed, predicted_labels_renamed, output_dict=True)
confusion_mat = confusion_matrix(y_test_renamed, predicted_labels_renamed)

# %%
# Save the classification results to a dictionary with the new labels
classification_results_renamed = {
    "Testing Accuracy": test_accuracy,
    "Training Accuracy": train_accuracy,
    "Epoch": 300,
    "Training Time (seconds)": training_time,
    "Precision": classification_rep
}

# Save the classification results to JSON files
with open("hasil_klasifikasi_bpnn_original.json", "w") as original_file:
    json.dump(classification_results, original_file, indent=4)

with open("hasil_klasifikasi_bpnn_renamed.json", "w") as renamed_file:
    json.dump(classification_results_renamed, renamed_file, indent=4)

# %%
# Visualize the confusion matrix with the new labels
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print(f"Testing Accuracy: {test_accuracy}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Training Time: {training_time} seconds")

# %%
# Retrieve training and validation loss and accuracy from the training history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Create a range of epoch numbers for the x-axis
epochs = range(1, len(training_loss) + 1)

# Calculate the classification report for all classes
classification_report_overall = classification_report(y_test_renamed, predicted_labels_renamed, output_dict=True)

# %%
# Extract F1 score, precision, and recall for all classes
f1_score_overall = classification_report_overall['macro avg']['f1-score']
precision_overall = classification_report_overall['macro avg']['precision']
recall_overall = classification_report_overall['macro avg']['recall']

# Print the overall metrics
print(f"Overall F1 Score: {f1_score_overall}")
print(f"Overall Precision: {precision_overall}")
print(f"Overall Recall: {recall_overall}")