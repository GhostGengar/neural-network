from numpy import dot, array, exp, random

class NeuralNetwork():
	def __init__(self):
		self.synaptic_weights = 2 * random.random((4, 1)) - 1

	# the sigmoid function normalise the pridicted outputs to 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# train the neural network	
	def train(self, given_inputs, given_outputs, training_times):
		for training in xrange(training_times):
			outputs = self.think(given_inputs)
			error = given_outputs - outputs
			adjustment = dot(given_inputs.T, error * self.__sigmoid_derivative(outputs))
			self.synaptic_weights += adjustment

		

	# the neural network thinks
	def think(self, some_inputs):
		return self.__sigmoid(dot(some_inputs, self.synaptic_weights))



if __name__ == '__main__':
	# initialise the neural network
	neural_network = NeuralNetwork()

	# given a set of inputs
	given_inputs = array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 0]])

	# given the corresponding outputs
	given_outputs = array([[1, 1, 0, 1, 0]]).T

	# how many time should the network learn
	training_times = 100000

	# print out synaptic weights before training
	print 'Synaptic weights before training is'
	print neural_network.synaptic_weights

	# train the neural network with the given datas
	neural_network.train(given_inputs, given_outputs, training_times)
	print 'training the network {0} time(s)'.format(training_times)

	# print the synaptic weights after training
	print 'Synaptic weights after training is'
	print neural_network.synaptic_weights

	# predict the value for a given set of input
	print 'Predict the output for [1, 0, 0, 0] ->'
	print neural_network.think(array([[1, 0, 0, 0]])) # it's 0