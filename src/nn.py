import random

import numpy

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # initialises an array with the biases
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        # initialises an array with the weights
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        
    def feedforward(self, inp):
        """ Calculates the guess of the network. For each layer of the network we multiply an array with the weights of all connections with a vector of the results of the previos layers. Additionally we add the biases. In the end we sigmoid the result, in order to receive a value between 0 and 1."""
        for b, w in zip(self.biases, self.weights):
            inp = sigmoid(numpy.dot(w, inp) + b)
        return inp

    
    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train our network with the given parameters """
        if test_data : l_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # shuffle our training set
            random.shuffle(training_data)
            # this creates a list of subsets of the training_data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_step(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), l_test)
            else:
                print "Epoch {0} complete".format(j)
                
    def train_step(self, mini_batch, eta):
        """ Train with a single mini batch. Update the weiths and biases by applying gradient descent. """
        # creates empty arrays with same shape as the weights and biases
        # representing the total gradients for the weights and biases
        gradients_b = [numpy.zeros(b.shape) for b in self.biases]
        gradients_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # returns the gradients for the weights and biases for one mini_batch
            delta_gradient_b, delta_gradient_w = self.backprob(x, y)
            # we add up all gradients from the mini_batches to one final gradient array which will be applied to the network
            gradients_b = [gb + dgb for gb, dgb in zip(gradients_b, delta_gradient_b)]
            gradients_w = [gw + dgw for gw, dgw in zip(gradients_w, delta_gradient_w)]

        # we apply our gradient to our weights and biases
        # we additionaly change the impact of our gradient by multiplying it with a value depending on the batch size and eta (change factor)
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, gradients_w)]
        self.biases = [b- (eta/len(mini_batch))*nb for b, nb in zip(self.biases, gradients_b)]
        

    def backprob(self, x, y):
        """ Returns the gradients for the weights and biases as an array"""
        gradient_b = [numpy.zeros(b.shape) for b in self.biases]
        gradient_w = [numpy.zeros(w.shape) for w in self.weights]

        # forward propagation
        #final result (after sigmoid)
        activation = x
        activations = [x]
        # vektor before sigmoid
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) +b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward propagation
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp= sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            gradient_b[-l] = delta
            gradient_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (gradient_b, gradient_w)

    def evaluate(self, test_data):
        test_results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(v):
    return 1.0/(1.0+numpy.exp(-v))

# derivative of the sigmoid fucntion
# = ableitung
def sigmoid_prime(v):
    return sigmoid(v) * (1 - sigmoid(v))
        
        
