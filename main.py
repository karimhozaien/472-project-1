from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Set the plot to grayscale for better visibility
plt.gray()

# Loop through all the images in the dataset
for index, (image, label) in enumerate(zip(digits.images, digits.target)):
    # Create a new figure for each digit
    plt.matshow(image)  # Display the digit image
    plt.title(f"Digit: {label}")  # Set the title to the digit's label
    plt.show()  # Show the plot