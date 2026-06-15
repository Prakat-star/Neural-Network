"""
mnist_svm with data visualization
"""

import mnist_loader
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def show_samples(images, labels, n=10):
    """
    Display n sample images from the dataset
    """
    for i in range(n):
        image = images[i].reshape(28, 28)  # MNIST images are 28x28
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
        plt.show()


def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    # Show some images from the dataset
    print("Showing sample images from MNIST dataset...")
    show_samples(training_data[0], training_data[1], 5)

    # Train model
    clf = LinearSVC()
    clf.fit(training_data[0], training_data[1])

    # Test model
    predictions = clf.predict(test_data[0])
    num_correct = sum(p == y for p, y in zip(predictions, test_data[1]))

    print("SVM classifier results:")
    print(f"{num_correct} of {len(test_data[1])} values correct")


if __name__ == "__main__":
    svm_baseline()