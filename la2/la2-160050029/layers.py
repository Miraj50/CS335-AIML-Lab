import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE

		self.data = np.matmul(X, self.weights) + self.biases
		return sigmoid(self.data)

		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		de_din = derivative_sigmoid(self.data) * delta
		de_dw = np.matmul(np.transpose(activation_prev), de_din)
		de_dprev = np.matmul(de_din, np.transpose(self.weights))
		# Update
		self.weights = self.weights - lr * de_dw
		for i in range(n):
			self.biases = self.biases - lr * de_din[i]
		return de_dprev

		# OLD, SLOWER IMPLEMENTATION USING A FOR LOOP
		# de_dprev = np.zeros((n, self.in_nodes))
		# for i in range(n):
		# 	de_din = derivative_sigmoid(self.data[[i]]) * delta[[i]]
		# 	de_dw = np.matmul(np.transpose(activation_prev[[i]]), de_din)
		# 	de_dprev[i] = np.transpose(np.matmul(self.weights, np.transpose(de_din)))
		# 	# Update
		# 	self.weights = self.weights - lr * de_dw
		# 	self.biases = self.biases - lr * de_din
		# return de_dprev

		# raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE

		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for i in range(n): # element in batch
			for j in range(self.out_depth): # no. of filters
				for y in range(self.out_row): # y -> row
					for x in range(self.out_col): # x -> column (For every row, iterate columns 0 to self.out_col)
						_x, _y = x*self.stride, y*self.stride
						t = np.sum(X[i, :, _y:_y+self.filter_row, _x:_x+self.filter_col] * self.weights[j]) + self.biases[j]
						self.data[i, j, y, x] = t
		return sigmoid(self.data)

		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		new_delta = np.zeros((n, self.in_depth, self.in_row, self.in_col))
		de_do = derivative_sigmoid(self.data) * delta
		for i in range(n): # element in batch
			for j in range(self.out_depth): # one filter
				for y in range(self.out_row):
					for x in range(self.out_col):
						_x, _y = x*self.stride, y*self.stride
						new_delta[i,:,_y:_y+self.filter_row,_x:_x+self.filter_col] += self.weights[j] * de_do[i,j,y,x]
						self.weights[j] -= lr * activation_prev[i,:,_y:_y+self.filter_row,_x:_x+self.filter_col] * de_do[i,j,y,x]
		self.biases -= lr * np.sum(de_do, axis=(0,2,3))
		return new_delta

		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE

		total = self.filter_row * self.filter_col
		temp = np.zeros((n, self.out_depth, self.out_row, self.out_col))

		for i in range(n):
			for j in range(self.out_depth):
				for y in range(self.out_row):
					for x in range(self.out_col):
						_x, _y = x*self.stride, y*self.stride
						temp[i, j, y, x] = np.mean(X[i, j, _y:_y+self.filter_row, _x:_x+self.filter_col])
		return temp

		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		temp = np.zeros(activation_prev.shape)
		temp[:,:,:self.stride*delta.shape[2],:self.stride*delta.shape[3]] = np.repeat(np.repeat(delta, self.stride, axis=2), self.stride, axis=3)/(self.filter_row*self.filter_col)
		return temp

		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
