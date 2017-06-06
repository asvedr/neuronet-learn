import numpy as np
import os
import random

class Neuron:
	def __init__(self, input_count):
		self.weights = np.array([random.random() * 2 - 0.99 for _ in range(input_count)], dtype=float)
	def predict(self, data):
		assert len(self.weights) == len(data)
		return (self.weights * data).sum() >= 0
	def predict_num(self, data):
		return (self.weights * data).sum()
	def fit(self, X, rate, Y):
		# fit rule
		# w[j] = w[j] + Dw[j] where Dw[j] = n(y(i) - _y(i))x(i)[j]
		# n     - learn rate [0 .. 1]
		# y(i)  - correct answer for 'i' sample
		# _y(i) - predict for 'i' sample
		# x(i)  - input for 'i' sample

		# XXX try to sum values for one set AND THEN save it as fit
		_Y = int(self.predict(X))
		Y = int(Y)
		for j in range(len(self.weights)):
			Dw = (Y - _Y) * X[j] * rate
			self.weights[j] = self.weights[j] + Dw

class Stat:
	def __init__(self):
		self.bad_yes = 0
		self.bad_yes_log = []
		self.bad_no  = 0
		self.bad_no_log = []
		self.total   = 0
	def __str__(self):
		return str({'bad_yes': self.bad_yes, 'bad_no': self.bad_no, 'total': self.total})
	def __repr__(self):
		return self.__str__()

class Predictor:
	def __init__(self):
		self.neurons = [
				('Iris-setosa', Neuron(4)),
				('Iris-versicolor', Neuron(4)),
				('Iris-virginica', Neuron(4))
			]
	def fit(self, data_set, set_name, rate):
		for neuron_name,neuron in self.neurons:
			neuron.fit(data_set, rate, set_name == neuron_name)
	def predict(self, data):
		for name,neuron in self.neurons:
			if neuron.predict(data):
				return name
	def statistics(self, dataset):
		stat = [(name,neur,Stat()) for name,neur in self.neurons]
		for set in dataset:
			X = set[:4]
			Y = set[4]
			for name,neuron,stobj in stat:
				pred = neuron.predict(X)
				if name == Y:
					if not pred:
						stobj.bad_no += 1
						stobj.bad_no_log.append(neuron.predict_num(X))
				else:
					if pred:
						stobj.bad_yes += 1
						stobj.bad_yes_log.append(neuron.predict_num(X))
				stobj.total += 1
		return [(obj[0], obj[2]) for obj in stat]

def get_set():
	def read(t):
		try:
			return float(t)
		except:
			return t
	with open(__file__[:-2] + 'csv', 'rt') as header:
		return [list(map(read, line.split(','))) for line in header.read().split('\n')]

def split_by_percent(list, percent_in_first):
	set = list[:]
	first = []
	for _ in range(int(len(set) * percent_in_first)):
		i = random.randint(0, len(set) - 1)
		first.append(set[i])
		del set[i]
	return (first, set)

predictor = Predictor()
(fit, check) = split_by_percent(get_set(), 0.5)
for _ in range(10):
	for seq in fit:
		predictor.fit(seq[:4], seq[4], 1)
fit_ok   = 0
check_ok = 0
# print('FIT')
for seq in fit:
	pred = predictor.predict(seq[:4])
	if pred == seq[4]:
		fit_ok += 1
	# print('X: %s, ANSWER: %s' % (seq, pred))
# print('CHECK')
for seq in check:
	pred = predictor.predict(seq[:4])
	if pred == seq[4]:
		check_ok += 1
	# print('X: %s, ANSWER: %s' % (seq, pred))
print('on fit %s of %s (%s%%)' % (fit_ok, len(fit), float(fit_ok) / len(fit)))
print('on check %s of %s (%s%%)' % (check_ok, len(check), float(check_ok) / len(check)))
for stat in predictor.statistics(fit + check):
	print(stat)
	# print('BY', stat[1].bad_yes_log)
	# print('BN', stat[1].bad_no_log)