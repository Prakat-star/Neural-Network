import mnist_loader
import network
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Test first 10 digits after training
for i in range(10):
    x, y = list(test_data)[i]
    output = net.feedforward(x)
    prediction = np.argmax(output)
    print(f"Test #{i}: True label={y}, Prediction={prediction}")

    image = np.reshape(x, (28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {prediction}, True: {y}")
    plt.axis('off')
    plt.show()