import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.weights = [np.random.randn(next_layer, current_layer) for current_layer, next_layer in zip(layer_size[:-1], layer_size[1:])]
        self.biases = [np.random.randn(size, 1) for size in layer_size[1:]]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def predict(self, input_data): # used for plotting only
        activation = input_data
        for bias, weight in zip(self.biases, self.weights):
            activation = self.sigmoid(np.dot(weight, activation) + bias)
        return activation

    def feed_foward(self, input_data):
        activation = input_data
        activations = [input_data]
        weighted_inputs = []
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = np.dot(weight, activation) + bias 
            weighted_inputs.append(weighted_input)
            activation = self.sigmoid(np.dot(weight, activation) + bias) # f(x)@w+b
            activations.append(activation)
        return activations, weighted_inputs
    
    def cost_derivative(self, output_activations, target):
        return (output_activations - target)

    def back_propagationn(self, input_data, target):
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights] 
        
        activations, weighted_inputs = self.feed_foward(input_data)  

        delta = self.cost_derivative(activations[-1], target) * self.sigmoid_derivative(weighted_inputs[-1]) 

        gradient_biases[-1] = delta
        gradient_weights[-1] = np.dot(delta, activations[-2].T) 
   
        for layer in range(2, len(self.layer_size)):
            weighted_input = weighted_inputs[-layer]
            delta = np.dot(self.weights[-layer + 1].T, delta) * self.sigmoid_derivative(weighted_input)
            gradient_biases[-layer] = delta
            gradient_weights[-layer] = np.dot(delta, activations[-layer - 1].T)

        return (gradient_biases, gradient_weights)

    def update_weights_biases(self, mini_batch, learning_rate):
        sum_gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        sum_gradient_weights = [np.zeros(weight.shape) for weight in self.weights]

        for input_data, target in mini_batch:
            delta_gradient_biases, delta_gradient_weights = self.back_propagationn(input_data, target)
            sum_gradient_biases = [sum_bias + delta_bias for sum_bias, delta_bias in zip(sum_gradient_biases, delta_gradient_biases)]
            sum_gradient_weights = [sum_weight + delta_weight for sum_weight, delta_weight in zip(sum_gradient_weights, delta_gradient_weights)]

        self.weights = [weight - (learning_rate / len(mini_batch)) * sum_weight for weight, sum_weight in zip(self.weights, sum_gradient_weights)]
        self.biases = [bias - (learning_rate / len(mini_batch)) * sum_bias for bias, sum_bias in zip(self.biases, sum_gradient_biases)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate):
        for _ in range(epochs):
            mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights_biases(mini_batch, learning_rate)

# Generate Data for Training
def generate_data(modes, samples_per_mode, means, variances):
    samples = []
    for _ in range(samples_per_mode):
        mode = np.random.randint(0, modes)
        mean = means[mode]
        variance = variances[mode]
        sample = np.random.normal(loc=mean, scale=variance)
        samples.append(sample)
    return samples

# Prepare Data for Neural Network Training
def prepare_training_data(class_0_samples, class_1_samples):
    combined_samples = class_0_samples + class_1_samples
    labels = [[1, 0]] * len(class_0_samples) + [[0, 1]] * len(class_1_samples)
    formatted_data = [(np.reshape(np.array(sample), (2, 1)), np.reshape(np.array(label), (2, 1)))
                     for sample, label in zip(combined_samples, labels)]
    return formatted_data

# Plot Decision Boundary of Neural Network
def plot_decision_boundary(network, training_data, class_0_count, class_1_count):
    x_values = [sample[0][0][0] for sample in training_data]
    y_values = [sample[0][1][0] for sample in training_data]
    x_min, x_max = min(x_values) - 1, max(x_values) + 1
    y_min, y_max = min(y_values) - 1, max(y_values) + 1

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    predictions = [network.predict(np.array(point).reshape((2, 1))) for point in grid_points]
    predictions = np.argmax(np.array(predictions), axis=1).reshape(grid_x.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(grid_x, grid_y, predictions, alpha=0.8, cmap="coolwarm")
    plt.scatter([sample[0][0] for sample in training_data[:class_0_count]], 
                [sample[0][1] for sample in training_data[:class_0_count]], 
                c="blue", label="Class 0")
    plt.scatter([sample[0][0] for sample in training_data[-class_1_count:]], 
                [sample[0][1] for sample in training_data[-class_1_count:]], 
                c="red", label="Class 1")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Neural Network Decision Boundary")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt, bbox_inches='tight')

# Streamlit Application Setup
st.title("Neural Network Decision Boundary Visualization")
st.sidebar.header("Network Parameters")

# User Inputs for Network Parameters and Data Generation
num_modes_class = st.sidebar.slider("Modes per Class", 1, 10, 2)
samples_per_mode_class = st.sidebar.slider("Samples per Mode", 10, 1000, 100)
number_of_hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 3, 3)
neurons_per_layer = st.sidebar.slider("Number of Neurons per Layer", 1, 10, 5)

# Adjust these values to increase the separation between class 0 and class 1
means_class_0 = [[-2, 0], [-2, 0]] 
variances_class_0 = [[0.5, 0.5], [0.5, 0.5]]  
means_class_1 = [[4, 2], [2, 4]]  
variances_class_1 = [[0.5, 0.5], [0.5, 0.5]]  

# Generating Data
class_0_data = generate_data(num_modes_class, samples_per_mode_class, means_class_0, variances_class_0)
class_1_data = generate_data(num_modes_class, samples_per_mode_class, means_class_1, variances_class_1)

training_data = prepare_training_data(class_0_data, class_1_data)

# Neural Network Initialization and Training
layer_size = [input_layer_size:=2] + [neurons_per_layer] * number_of_hidden_layers + [output_layer_size:=2]
network = NeuralNetwork(layer_size)
network.train(training_data, epochs=300, mini_batch_size=10, learning_rate=0.01)

# Plotting Decision Boundary
plot_decision_boundary(network, training_data, samples_per_mode_class, samples_per_mode_class)
