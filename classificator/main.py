#!/usr/bin/env python3
import numpy as np
import random
import argparse
import os
import sys

class Neuron:
	def __init__(self, input_count, name):
		self.weights = np.array([random.random() * 2 - 0.99 for _ in range(input_count)], dtype=float)
		self.name = name
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

		_Y = int(self.predict(X))
		Y = int(Y)
		for j in range(len(self.weights)):
			Dw = (Y - _Y) * X[j] * rate
			self.weights[j] = self.weights[j] + Dw
	def save(self):
		return list(self.weights)
	def load(self, lst):
		self.weights = np.array(lst)		

class CantSplitError(Exception):
	"""docstring for CantSplitError"""
	def __init__(self):
		super().__init__()

class Classificator(object):
	'''
		fit set: [([x1, x2, ..., xi], Y)]
		self.neurons : [neuron1, neuron2, ...]
			algo:
				if neuron1 then neuron1.name else if neuron2 then neuron2.name else ...
				use CONSEQUENTALY!!!
	'''
	def __init__(self):
		self.neurons = []
		# if no one neuron say 'YES' we use .rest_group as answer
		self.rest_group = None

	def predict(self, data):
		for neuron in self.neurons:
			if neuron.predict(data):
				return neuron.name
		return self.rest_group

	def save(self):
		return {
			'neurons': [{'N': n.name, 'W': n.save()} for n in self.neurons],
			'rest' : self.rest_group,
			'pcount' : len(self.neurons[0].weights)
		}

	@classmethod
	def load(cls, jsconf):
		self = cls()
		pcount = jsconf['pcount']
		self.rest_group = jsconf['rest']
		self.neurons = []
		for n in jsconf['neurons']:
			neuron = Neuron(pcount, n['N'])
			neuron.load(n['W'])
			self.neurons.append(neuron)
		return self

	@classmethod
	def from_fitset(cls, fitset):
		'''
			fitset: [([X1, ... Xi], Y)]
			Xi - params
			Y - class
		'''
		self = cls()
		self.neurons    = []
		self.rest_group = None

		random.seed()

		def make_fit_and_check(original_dict):
			percent = 0.6
			fit   = []
			check = []
			for X in original_dict.values():
				random.shuffle(X)
				split = int(len(X) * percent)
				fit.extend(X[:split])
				check.extend(X[split:])
			random.shuffle(fit)
			random.shuffle(check)
			return (fit,check)

		# because last value is Y
		paramcount = len(fitset[0]) - 1

		def split_by_Y(fitset, Y):
			allow_accuracy = 0.9

			neuron         = Neuron(paramcount, Y)
			snapshot       = None
			last_good_rate = 0
			total          = len(fitset)
			for _ in range(100):
				# fit
				for set in fitset:
					neuron.fit(set[0], 1, set[1] == Y)
				# check
				success = len([1 for set in fitset if neuron.predict(set[0]) == (set[1] == Y)])
				if success == total:
					return neuron
				if success > last_good_rate:
					snapshot = neuron.save()
					last_good_rate = success
			if float(last_good_rate) / total > allow_accuracy:
				neuron.load(snapshot)
				return neuron
			else:
				return None

		def final_check(fit, check):
			percent = lambda l1, l2: int(100 * float(len(l1)) / len(l2))
			print('FIT:')
			f_fit = [(set[0],set[1],self.predict(set[0])) for set in fit if self.predict(set[0]) != set[1]]
			print('%s of %s (%s%%)' % (len(f_fit), len(fit), percent(f_fit, fit)))
			f_che = [(set[0],set[1],self.predict(set[0])) for set in check if self.predict(set[0]) != set[1]]
			print('%s of %s (%s%%)' % (len(f_che), len(check), percent(f_che, check)))
			return percent(f_fit, fit) < 10 and percent(f_che, check) < 10


		all_names = set(map(lambda a: a[-1], fitset))
		grouped = {}
		for name in all_names:
			grouped[name] = []
		for item in fitset:
			grouped[item[-1]].append( (item[:-1], item[-1]) )

		for full_fit_try in range(5):
			(fit_orig,check) = make_fit_and_check(grouped)
			fit              = fit_orig[:]
			names            = all_names.copy()
			self.neurons     = []
			try:
				while len(names) != 1:
					neuron = None
					for name in names:
						neuron = split_by_Y(fit, name)
						if neuron:
							break
					if neuron is None:
						raise CantSplitError()
					print('split by', neuron.name)
					names.remove(neuron.name)
					fit = [s for s in fit if s[1] != neuron.name]
					self.neurons.append(neuron)
				self.rest_group = list(names)[0]
				# success branch
				# all splitted by neurons
				# make final check and end
				if final_check(fit_orig, check):
					return self
				else:
					print('final check fail')
			except CantSplitError:
				continue
		# bad branch
		# net has errors in some split
		print(self.neurons)
		self.neurons = []
		raise CantSplitError

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='fit by set')
parser.add_argument('-s', help='save to file(only for fit)')
parser.add_argument('-p', help='predict by params')
parser.add_argument('-l', help='load from file(only for predict)')
args = vars(parser.parse_args())
def make(t):
	try:
		return float(t)
	except:
		return t
if args['f']:
	if args['s'] is None:
		print('no out file')
		sys.exit(1)
	fit = [[make(t) for t in line.split(',')] for line in open(args['f'], 'rt').read().split('\n')]
	with open(args['s'], 'wt') as out:
		snapshot = Classificator.from_fitset(fit).save()
		out.write(str(snapshot))
		out.write('\n')
	print('OK')
elif args['p']:
	if args['l'] is None:
		print('no input file')
		sys.exit(1)
	conf = eval(open(args['l'], 'rt').read())
	ans = Classificator.load(conf).predict([make(s) for s in args['p'].split(',')])
	print(ans)
else:
	print('predict or fit?')