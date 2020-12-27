import numpy as np
import pandas as pd

df = pd.read_csv('ex1data1.txt', header=None)

theta = [0, 0]


def hypothesis(theta, x):
	return theta[0] + theta[1] * x

def cost_function(theta, x, y):
	return (1/2*m) * np.sum((hypothesis(theta, x)- y)**2)

m = len(df)

def gradient_descent(theta, x, y, epoch, alpha):
	cost = []
	i = 0
	while i < epoch:
		hx = hypothesis(theta, x)
		theta[0] -= alpha * (sum(hx-y)/m)
		theta[1] -= (alpha * (np.sum(hx-y)*x))/m
		cost.append(cost_function(theta, x, y))
		i += 1
	return theta, cost

def predict(theta, x, y, epoch, alpha):
	theta, cost = gradient_descent(theta, x, y, epoch, alpha)
	return hypothesis(theta, x), cost, theta

y_predict, cost, theta = predict(theta, df[0], df[1], 2000, 0.01)
print(theta)