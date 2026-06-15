import mnist_loader
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)

# show first 10 digits
for i in range(10):

    x, y = training_data[i]

    image = np.reshape(x, (28, 28))
    label = np.argmax(y)

    plt.subplot(2,5,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(label)
    plt.axis("off")

plt.show()