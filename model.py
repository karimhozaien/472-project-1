# Alessandro Condina ID:40158684, Karim Hozaien (ID: 40349984)

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


def train_and_evaluate():
    # Load digits dataset images
    digits = load_digits()
    raw_data = digits.data  # Un-normalized data
    label = digits.target  # Actual labels (numbers 0–9)

    print("Data instances and attributes:", raw_data.shape)
    print("Label instances:", label.shape)
    unique, counts = np.unique(label, return_counts=True)
    total_count = 0
    for digit, count in zip(unique, counts):
        print(f"Label {digit}, count {count}")
        total_count += count
    print(f"Total count: {total_count}")

    # Numpy exploration on raw data
    print("\nUn-Normalized data statistics:")
    print("Pixel mean:", np.mean(raw_data))
    print("Pixel std deviation:", np.std(raw_data))
    print("Max pixel value:", np.max(raw_data))
    print("Min pixel value:", np.min(raw_data))

    data = digits.data / 16.0  # Normalize the pixel values

    # Numpy exploration on normalized data
    print("\nNormalized data statistics:")
    print("Pixel mean:", np.mean(data))
    print("Pixel std deviation:", np.std(data))
    print("Max pixel value:", np.max(data))
    print("Min pixel value:", np.min(data))

    # Split the data 80% for training 20% for testing
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=1)

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(data_train, label_train)

    # Predict on the test set
    predictions = model.predict(data_test)

    # Print model evaluation
    print("\nAccuracy:", accuracy_score(label_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(label_test, predictions))
    print("\nClassification Report:\n", classification_report(label_test, predictions))


if __name__ == "__main__":
    train_and_evaluate()
