# Adaptive Linear Neuron
# weight update formula
# w = w + delta w
# delta w{j} = -u * sum{i}( (y{i} - f(z{i})) * x{i}[j] )

# where
#	y - actial value
#	f(z(...)) current continious value f(z(...)) = sum[i](w[i] * x[i])
#	u - learn rate
import numpy as np
import random
import loger

# numpy.array.dot(a,b) => (a * b).sum()

class Disconvergence(Exception):
	def __init__(self):
		super().__init__(self, 'disconv')

class Neuron(object):
	"""docstring for Neuron"""
	def __init__(self, weight_count, name):
		super().__init__()
		self.weights = np.zeros(weight_count)
		self.w0 = 0.0
		self.name = name
	def predict(self, data):
		return self.predict_value(data) > 0
	def predict_value(self, data):
		return self.weights.dot(data) + self.w0
	def fit(self, samples, allow_cost=0.001, max_iter=-1):
		# on u = 0.01 fit is gonna bad, but on 0.001 it gonna ok
		u = 0.001
		X = []
		Y = []
		for samp in samples:
			X.append(samp.X)
			Y.append(samp.Y)
		X = np.array(X)
		Y = np.array(Y)
		self.cost    = []
		self.weights *= 0
		self.w0      = 0
		last_cost = allow_cost + 1
		fit_iter_count = 0
		while last_cost > allow_cost and (fit_iter_count < max_iter or max_iter < 0):
			out = np.array(list(map(self.predict_value, X)))
			errors = Y - out
			self.w0 += u * errors.sum() # why sum all into w0 ?
			self.weights += X.T.dot(errors) * u # T is transpose. Why transpose ?
			self.cost.append((errors ** 2).sum() / 2.0)
			last_cost = self.cost[-1]
			fit_iter_count += 1
			try:
				convergence = self.cost[-1] < self.cost[-3]
			except:
				convergence = True
			if not convergence:
				raise Disconvergence()
		return len(self.cost)

	def save(self):
		return [self.w0] + list(self.weights)
	def load(self, lst):
		self.w0 = lst[0]
		self.weights = np.array(lst[1:])

class Sample(object):
	def __init__(self, X, Y):
		self.X = np.array(X)
		self.Y = Y
	def to_num(self, tmpl):
		return Sample(self.X, 1 if self.Y == tmpl else -1)
	@staticmethod
	def to_num_class(tmpl):
		return lambda sample: sample.to_num(tmpl)

def test():
	n = Neuron(4, 'test')
	fit = np.array([
		Sample([5.1,3.5,1.4,0.2],1),
		Sample([4.9,3.0,1.4,0.2],1),
		Sample([4.7,3.2,1.3,0.2],1),
		Sample([7.0,3.2,4.7,1.4],-1),
		Sample([6.4,3.2,4.5,1.5],-1),
		Sample([6.9,3.1,4.9,1.5],-1),
		Sample([6.3,3.3,6.0,2.5],-1),
		Sample([5.8,2.7,5.1,1.9],-1),
		Sample([7.1,3.0,5.9,2.1],-1)
	])
	check_y = [5.4,3.9,1.7,0.4]
	check_n = [6.3,3.3,4.7,1.6]
	n.fit(fit, max_iter=100)
	print(n.cost)
	#print([n.predict(s.X) for s in fit])
	print(n.predict(check_y))
	print(n.predict(check_n))

if __name__ == '__main__':
	test()