import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	newX = np.ones((X.shape[0], 1))
	for i in range(1, X.shape[1]):
		col = X[:,i]
		if isinstance(X[0][i], str):
			newX = np.c_[newX, one_hot_encode(col, np.unique(col))]
		else:
			newX = np.c_[newX, (col-np.mean(col))/np.std(col)]
	return newX.astype('float64'), Y.astype('float64')

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return Y+X@W+2*_lambda*W

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	Y_n = -np.transpose(X)@Y*2 # Precomputation
	X_n = np.transpose(X)@X*2 # Precomputation
	w = np.zeros((X.shape[1], 1))
	for _ in range(max_iter):
		grad = grad_ridge(w, X_n, Y_n, _lambda)
		w_n = w - lr*grad
		if np.linalg.norm(w_n-w, 2) < epsilon:
			break
		w = np.copy(w_n)
	return w

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	N = X.shape[0]
	step = N//k
	sse_list = []
	for _l in lambdas:
		s = 0.0
		for i in range(0, N, step):
			W = algo(np.r_[X[:i], X[i+step:]], np.r_[Y[:i], Y[i+step:]], _l) # Train
			s += sse(X[i:i+step], Y[i:i+step], W) # Test
		sse_list.append(s/k)
	return sse_list

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	XX = np.transpose(X)@X #Precomputation
	YX = np.transpose(X)@Y #Precomputation
	D = X.shape[1]
	w = np.zeros((D,1))
	for _ in range(max_iter):
		for i in range(D):
			w[i,0] = 0
			# c = 2*(np.transpose(Y-X@w)@X[:,i])[0] #Inefficient
			c = 2*(YX[i]-XX[i]@w)
			if c>_lambda:
				w[i,0] = (c-_lambda)/(2*np.sum(X[:,i]**2))
			elif c<-_lambda:
				w[i,0] = (c+_lambda)/(2*np.sum(X[:,i]**2))
			else:
				w[i,0] = 0.0
	# w1 = np.random.rand(D,1) # Checking for a different initialization
	# for _ in range(max_iter):
	# 	for i in range(D):
	# 		w1[i,0] = 0
	# 		c = 2*(YX[i]-XX[i]@w1)
	# 		if c>_lambda:
	# 			w1[i,0] = (c-_lambda)/(2*np.sum(X[:,i]**2))
	# 		elif c<-_lambda:
	# 			w1[i,0] = (c+_lambda)/(2*np.sum(X[:,i]**2))
	# 		else:
	# 			w1[i,0] = 0.0
	# k=0
	# for i in range(D):
	# 	if w[i,0] == w1[i,0]==0:
	# 		k+=1
	# print(k)
	return w

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	#Ridge
	# lambdas = [12, 12.1, 12.2, 12.3, 12.4, 12.43, 12.5, 12.6, 12.7, 12.8, 12.9, 13] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	# print("Minimum SSE = {} for {}".format(min(scores), lambdas[scores.index(min(scores))]))
	#Co-ord
	# lambdas = [i for i in range(300000, 450000, 5000)]
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	# print("Minimum SSE = {} for {}".format(min(scores), lambdas[scores.index(min(scores))]))
	# plot_kfold(lambdas, scores)