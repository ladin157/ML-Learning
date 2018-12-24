"""
https://medium.com/deep-math-machine-learning-ai/chapter-7-1-neural-network-from-scratch-in-python-b880b0ff5f7b
https://github.com/Madhu009/Deep-math-machine-learning.ai/blob/master/Neural_Networks/Artificial-Neural-Network_from_scratch.ipynb
"""
import numpy as np
from IPython.display import Image, display
import matplotlib.pyplot as plt


# print("Hello world")

XORdata = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
X = XORdata[:, 0:2]
y = XORdata[:, -1]


# print(X)

def print_network(net):
    for i, layer in enumerate(net, 1):
        print("Layer: {} ".format(i))
        for j, neuron in enumerate(layer, 1):
            print("Neuron {}: ".format(j), neuron)


def initialize_network():
    input_neurons = len(X[0])
    # print(input_neurons)
    hidden_neurons = input_neurons + 1
    output_neurons = 2

    n_hidden_layers = 1

    net = list()

    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])
        hidden_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
        net.append(hidden_layer)

    output_layer = [{'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)
    return net


net = initialize_network()

# display(Image("img/network.jpg"))
print_network(net=net)


def activate_sigmoid(sum):
    return 1 / (1 + np.exp(-sum))


def forward_propagation(net, input):
    row = input
    for layer in net:
        prev_input = np.array([])
        for neuron in layer:
            sum = neuron['weights'].T.dot(row)

            result = activate_sigmoid(sum=sum)
            neuron['result'] = result

            prev_input = np.append(prev_input, [result])
        row = prev_input
    return row


def sigmoid_derivative(output):
    return output * (1.0 - output)


def back_propagation(net, row, expected):
    for i in reversed(range(len(net))):
        layer = net[i]
        errors = np.array([])
        if i == len(net) - 1:
            results = [neuron['result'] for neuron in layer]
            errors = expected - np.array(results)
        else:
            for j in range(len(layer)):
                herror = 0
                nextlayer = net[i + 1]
                for neuron in nextlayer:
                    herror += (neuron['weights'][j] * neuron['delta'])
                errors = np.append(errors, [herror])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['result'])


def update_weights(net, input, lrate):
    for i in range(len(net)):
        inputs = input
        if i != 0:
            inputs = [neuron['result'] for neuron in net[i - 1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]


def training(net, epochs, lrate, n_outputs):
    errors = []
    for epoch in range(epochs):
        sum_error = 0
        for i, row in enumerate(X):
            outputs = forward_propagation(net, row)
            exptected = [0.0 for i in range(n_outputs)]
            exptected[y[i]] = 1

            sum_error += sum([(exptected[j] - outputs[j]) ** 2 for j in range(len(exptected))])
            back_propagation(net, row, exptected)
            update_weights(net, row, lrate)

        if epoch % 10000 == 0:
            print('>epoch=%d,error=%.3f' % (epoch, sum_error))
            errors.append(sum_error)
    return errors


errors = training(net=net, epochs=100000, lrate=0.05, n_outputs=2)


# epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# plt.plot(epochs, errors)
# plt.xlabel("epochs in 10000's")
# plt.ylabel('error')
# plt.show()

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(net, row)
    return outputs


pred = predict(net, np.array([1, 0]))
output = np.argmax(pred)
print(output)

print_network(net)
