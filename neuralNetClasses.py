import math
import numpy as np

class Connector:
	def __init__(self, neuronConnection):
		self.neuronConnection = neuronConnection
		self.weight = np.random.normal()
		self.deltaWeight = 0.

class Neuron:
	alpha = 0.01
	eta = 0.001

	def __init_(self, layer):
		self.output = 0.
		self.error = 0.
		self.gradient = 0.
		self.dendrons = []
		if layer is None:
			pass
		else:
			for neuron in layer:
				connection = Connector(neuron)
				self.dendrons.append(connection)

	#accumulates error of neurons during backpropogation
	def addError(self, error):
		self.error = self.error + error

	#getters and setters for each var in Neuron
	#some setters/getters are not needed, but included for full class control
	def getError(self):
		return self.error

	def setError(self, error):
		self.error = error

	def getOutput(self):
		return self.output

	def setOutput(self, output):
		self.output = output

	def getGradient(self):
		return self.gradient

	def setGradient(self, gradient):
		self.gradient = gradient

	#activation function
	def sigmoid(self, x):
        return 1./(1. + math.exp(-x * 1.))

    #derivation of sigmoid function
    def derSigmoid(self, x):
        return x * (1.0 - x)

    #function meant to check for neurons that are already connected
    #any connected neurons are multiplied by the dendrons' weight
    def feedForward(self):
        sum = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sum = sum + dendron.neuronConnection.getOutput() * dendron.weight
        self.output = self.sigmoid(sum)

    #backpropogation method used to calculate the gradient based on the weights of the dendrons
    def backpropogate(self):
        self.gradient = self.error * self.derSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.deltaWeight = self.eta * dendron.neuronConnection.output * self.gradient + self.alpha * dendron.deltaWeight
            dendron.weight = dendron.weight + dendron.deltaWeight
            dendron.neruonConnection.addError(dendron.weight * self.gradient)
        self.error = 0

class Network:
    #threshold variable for thresholded result output
    threshold = 0.5

    #sets up the network using a given topology as its input
    #topology is a set of numbers where the amount of numbers corresponds to the number of layers
    #each number represents the number of neurons in its respective layer
    #-1 sets up a bias neuron
    def __init__(self, topology):
        self.layers = []
        for neurons in topology:
            layer = []
            for i in range(neurons):
                if(len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    #establishes the inputs for each neuron in the first layer (input layer)
    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    #function for calculating the error based on the output and target value
    def getError(self, target):
        error = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            error = error + e ** 2
        error = error / len(target)
        error = math.sqrt(error)
        return error

    #apply forward feeding for the network
    def feedForward(self):
        for layer in self.layers[1:]:
            for neurons in layer:
                neurons.feedForward()

    #backpropogation step for the network; calls each neuron's backpropogation step during the step
    def backpropogate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::1]:
            for neuron in layer:
                neuron.backpropogate()

    #results output; one standard, one using threshold variable as an output threshold
    def getResults(self):
        output = []
        for neurons in self.layers[-1]:
            output.append(neurons.getOutput())
        #pop the bias neuron so it's not included in the output
        output.pop()
        return output

    def getThresholdedResults(self):
        output = []
        for neurons in self.layers[-1]:
            o = neurons.getOutput()
            if (o &gt; self.threshold):
                o = 1
            else:
                o = 0
            output.append(o)
        #pop the bias neuron so it's not included in the output
        output.pop()
        return output
