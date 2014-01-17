from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

try:
    import numpy as np
except:
    exit()

from deap import benchmarks
A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], 
[0.75, 0.25], [0.75, 0.75], [0.35, 0.35]]
C = [0.002, 0.005, 0.005, 0.005, 0.005, 0.005]


def schwefel_arg0(sol):
    return benchmarks.schwefel(sol)[0]
def h1_arg0(sol):
    return benchmarks.h1(sol)[0]
class FunctionObject:
	def __init__(self, xMin = -100, xMax = 100, yMin = -100, yMax = 100, step = 1):
		self.__function = self.shekel_arg0
		self.xMin = xMin
		self.xMax = xMax
		self.yMin = yMin
		self.yMax = yMax
		self.step = step
	def shekel_arg0(self, sol):
	    return benchmarks.shekel(sol, self.A, self.C)[0]
	def setExtremums(self, A, C):
		self.A = A
		self.C = C

	def calculateWithXY(self, x, y):
		return self.__function([x, y])
	def calculateWithYX(self, y, x):
		return self.__function([x, y])
	def calculate(self, cords):
		# print "Function: cords: "+str(cords)
		return self.__function(cords)

	def plot(self):
		fig = plt.figure()
		# ax = Axes3D(fig, azim = -29, elev = 50)
		ax = Axes3D(fig)
		X = np.arange(self.xMin, self.xMax, self.step)
		Y = np.arange(self.yMin, self.yMax, self.step)
		X, Y = np.meshgrid(X, Y)
		Z = np.zeros(X.shape)
		for i in xrange(X.shape[0]):
		    for j in xrange(X.shape[1]):
		        Z[i,j] = self.__function((X[i,j],Y[i,j]))

		ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
		 
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

