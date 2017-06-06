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
	def save(self):
		return list(self.weights)
	def load(self, lst):
		self.weights = np.array(lst)

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
	def fit_with_stat(self, datasets, minrate):
		count = 0
		last_good_val = None
		while True:
			for dataset in datasets:
				X = dataset[:4]
				Y = dataset[4]
				self.fit(X, Y, minrate)
			stat = self.statistics(datasets)
			bad_neurs  = 0
			fail_count = 0
			for i in range(len(stat)):
				stnew = stat[i][1]
				if stnew.bad_yes != 0 or stnew.bad_no != 0:
					bad.append(i)
					fail_count += stnew.bad_yes + stnew.bad_no
			print('iter %s outers %s fail count %s' % (i, bad, fail_count))
			if bad_neurs == 0 or count >= max_fit:
				if last_good_val:
					self.load(last_good_val)
				return
			count += 1
	def load(self, vals):
		for pair in zip(self.neurons, vals):
			pair[0][1].load(pair[1])
	def save(self):
		return [neur[1].save() for neur in self.neurons]

class N2:
	def __init__(self):
		self.setosa = Neuron(4)
		self.virginica = Neuron(4)
		self.sname = 'Iris-setosa'
		self.vname = 'Iris-virginica'
		self.other = 'Iris-versicolor'
	def fit(self, dataset):
		dataset = [(set[:4], set[4]) for set in dataset]
		max_fit = 1000	
		err_count = 1
		best_val = len(dataset)
		snapshot = None
		for i in range(max_fit):
			err_count = 0
			for set in dataset:
				if self.setosa.predict(set[0]) != (set[1] == self.sname):
					err_count += 1
				self.setosa.fit(set[0], 0.2, set[1] == self.sname)
			if err_count < best_val:
				snapshot = self.setosa.save()
				if err_count == 0:
					break
		self.setosa.load(snapshot)
		print('setosa ok')
		dataset = [set for set in dataset if set[1] != self.sname]
		err_count = 1
		best_val = len(dataset)
		snapshot = None
		for i in range(max_fit):
			err_count = 0
			for set in dataset:
				if self.virginica.predict(set[0]) != (set[1] == self.vname):
					err_count += 1
				self.virginica.fit(set[0], 0.2, set[1] == self.vname)
			if err_count < best_val:
				snapshot = self.virginica.save()
				if err_count == 0:
					break
		self.virginica.load(snapshot)
		print('virg ok')
		print({'s': self.setosa.save(), 'v': self.virginica.save()})
	def predict(self, data):
		if self.setosa.predict(data):
			return self.sname
		elif self.virginica.predict(data):
			return self.vname
		else:
			return self.other

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

predictor = N2()
(fit, check) = split_by_percent(get_set(), 0.5)
# predictor.fit_with_stat(fit, 0.01)
predictor.fit(fit)
print('CHECK FIT')
fok = len([1 for seq in fit if predictor.predict(seq[:4]) == seq[4]])
print('%s of %s (%s%%)' % (fok, len(fit), int(100 * float(fok) / len(fit)) ))
print('EXAM')
cok = len([1 for seq in check if predictor.predict(seq[:4]) == seq[4]])
print('%s of %s (%s%%)' % (cok, len(check), int(100 * float(cok) / len(check)) ))

'''
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
'''