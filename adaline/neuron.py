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
	def __init__(self, weight_count, name, u, use_normal=False):
		super().__init__()
		self.u = u # learn coefficent
		self.weights = np.zeros(weight_count)
		self.w0 = 0.0
		self.name = name
		self.use_normal = use_normal

	def predict(self, data):
		return self.predict_value(data) > 0

	def predict_value(self, data):
		if self.use_normal:
			temp = (data - self.means) / self.sigmas
			# cause w0 - mean(w0) is always 0
			return self.weights.dot(temp)# + self.w0
		else:
			return self.weights.dot(data) + self.w0

	def normalize(self, inputs):
		data = inputs.T
		sigmas = []
		means = []
		for i in range(len(data)):
			values = data[i]
			means.append(values.mean())
			sigmas.append(values.std())
		self.sigmas = np.array(sigmas)
		self.means  = np.array(means)

	def fit(self, samples, allow_cost=0.001, max_iter=-1):
		# on u = 0.01 fit is gonna bad, but on 0.001 it gonna ok
		u = self.u # 0.0001
		X = []
		Y = []
		for samp in samples:
			X.append(samp.X)
			Y.append(samp.Y)
		X = np.array(X)
		# normalization can make learning more efficent
		if self.use_normal:
			self.normalize(X)
		Y = np.array(Y)
		self.cost    = []
		self.weights *= 0
		self.w0      = 0
		last_cost = allow_cost + 1
		fit_iter_count = 0
		while last_cost > allow_cost and (fit_iter_count < max_iter or max_iter < 0):
			out = np.array(list(map(self.predict_value, X)))
			errors = Y - out
			# why sum all into w0 ? because x0 always 1
			self.w0 += u * errors.sum()
			# for every W{i} we make errors.sum * X{i} * u
			# T is transpose. Why transpose ? for one dismension transpose can be ignored
			self.weights += X.T.dot(errors) * u
			# cost is just for loging and detecting disconvergence
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
		res = {}
		res['use_normal'] = self.use_normal
		if self.use_normal:
			res['sigmas'] = self.sigmas
			res['means'] = self.means
		res['weights'] = [self.w0] + list(self.weights)
		return res

	def load(self, js):
		if js['use_normal']:
			self.means = js['means']
			self.sigmas = js['sigmas']
		lst = js['weights']
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
	n = Neuron(4, 'test', 0.001, False)
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
	try:
		n.fit(fit, max_iter=500)
	except Disconvergence:
		print('disconv')
		return
	print('fit count', len(n.cost))
	#print([n.predict(s.X) for s in fit])
	print(n.predict(check_y))
	print(n.predict(check_n))

if __name__ == '__main__':
	test()