import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import linear_model

def run():
	inp = np.loadtxt('data.txt',dtype='float', delimiter='	')
	n, d = inp.shape

	#number of points to generate
	points = 2
	mat = np.zeros((n*points,d))

	for i in range(n):
		for j in range(points):
			x = random.uniform(-1.0,1.0)
			mat[points*i+j,0] = x + inp[i,0]
			y = random.uniform(-1.0,1.0)
			mat[points*i+j,1] = y + inp[i,1]

	plt.scatter(mat[:,0],mat[:,1],color='red')


	reg = linear_model.LinearRegression()
	x_data = np.zeros((n*points,1))
	y_data = np.zeros((n*points,1))
	x_data[:,0]=mat[:,0]
	y_data[:,0]=mat[:,1]

	reg.fit(x_data, y_data)
	print('coefficient \n',reg.coef_)

	x_test = np.zeros((n,1))
	x_test[:,0] = inp[:,0]
	y_pred = reg.predict(x_test)
	plt.plot(x_test,y_pred,color='blue')
	plt.show()

if __name__ == '__main__': 
	run()