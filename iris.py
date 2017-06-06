import numpy as np
import os
import random

class Neuron:
	def __init__(self, input_count):
		self.weights = np.array([random.random() * 2 - 0.99 for _ in range(input_count)], dtype=float)
	def predict(self, data):
		assert len(self.weights) == len(data)
		return (self.weights * data).sum() >= 0
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

neuron = Neuron(4)
(fit, check) = split_by_percent(get_set(), 0.5)
name = 'Iris-setosa'
for _ in range(10):
	for seq in fit:
		neuron.fit(seq[:4], 1, seq[4] == name)
fit_ok   = 0
check_ok = 0
print('FIT')
for seq in fit:
	pred = neuron.predict(seq[:4])
	ok = (seq[4] == name)
	if pred == ok:
		fit_ok += 1
	print('X: %s, ANSWER: %s' % (seq, pred))
print('CHECK')
for seq in check:
	pred = neuron.predict(seq[:4])
	ok = seq[4] == name
	if pred == ok:
		check_ok += 1
	print('X: %s, ANSWER: %s' % (seq, pred))
print('on fit %s of %s' % (fit_ok, len(fit)))
print('on check %s of %s' % (check_ok, len(check)))
# test