# Alessandro Condina ID:40158684, Karim Hozaien (ID: 40349984)
# This lets you choose an image out of 1796 images to view using matplotlib
# The user is prompted to enter an index number, and an image at that index will appear

import matplotlib
matplotlib.use('TkAgg') # had to add this backend to render images
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

print("Dataset loaded.")
print(f"There are {len(digits.images)} images available (0 to {len(digits.images) - 1}).")

while True:
    try:
        index = input("\nEnter an image index to view (or 'exit' to quit): ")

        if index.lower() == 'exit':
            print("Exiting viewer.")
            break

        index = int(index)
        if 0 <= index < len(digits.images):
            plt.matshow(digits.images[index], cmap='gray')
            plt.title(f"Label: {digits.target[index]}")
            plt.show()
        else:
            print(f"Please enter a number between 0 and {len(digits.images) - 1}.")
    except ValueError:
        print("Invalid input. Please enter a number or 'exit'.")
