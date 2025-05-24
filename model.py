# Alessandro Condina ID:40158684

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_and_evaluate():
    # Load digits dataset images
    digits = load_digits()
    X = digits.data / 16.0  # Normalize the pixel values (original range 0-16)
    y = digits.target  # Actual labels (0–9)

    # Split the data: 80% for training 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))


if __name__ == "__main__":
    train_and_evaluate()
