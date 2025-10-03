import numpy as np
import tensorflow as tf

import numpy as np
from tf.keras.datasets import mnist

class VacuumTubeNeuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def activate(self, x):
        # Activation function: a simple step function
        return 1 if x > 0 else 0

    def forward(self, inputs):
        # Weighted sum of inputs + bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activate(weighted_sum)

# Example usage of the VacuumTubeNeuron:
neuron = VacuumTubeNeuron(3)
inputs = np.array([1.0, 0.5, -1.5])
output = neuron.forward(inputs)
print("Neuron output: ", output)

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize layers
        self.hidden_layer = [VacuumTubeNeuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [VacuumTubeNeuron(hidden_size) for _ in range(output_size)]

    def forward(self, inputs):
        # Forward pass through the hidden layer
        hidden_outputs = np.array([neuron.forward(inputs) for neuron in self.hidden_layer])
        # Forward pass through the output layer
        output = np.array([neuron.forward(hidden_outputs) for neuron in self.output_layer])
        return output

    def predict(self, inputs):
        # Get the output of the network and return the predicted class
        output = self.forward(inputs)
        return np.argmax(output)

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the images (flatten and normalize)
train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0

# Create a simple neural network with 28*28 input neurons, 128 hidden neurons, and 10 output neurons
input_size = 28 * 28
hidden_size = 128
output_size = 10
neural_net = SimpleNeuralNet(input_size, hidden_size, output_size)

# Test the neural network on a single image
test_image = test_images[0]
predicted_class = neural_net.predict(test_image)
print("Predicted class: ", predicted_class)

