import neuron as NeuronM
import os
import random
import numpy as np
import loger

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
			for samples in original_dict.values():
				random.shuffle(samples)
				split = int(len(samples) * percent)
				fit.extend(samples[:split])
				check.extend(samples[split:])
			random.shuffle(fit)
			random.shuffle(check)
			return (fit,check)

		paramcount = len(fitset[0].X)

		def split_by_Y(fitset, Y):
			allow_accuracy = 0.9

			neuron         = NeuronM.Neuron(paramcount, Y, 0.00005)
			# neuron         = NeuronM.Neuron(paramcount, Y, 0.000000001, True)
			snapshot       = None
			last_good_rate = 0
			total          = len(fitset)
			for _ in range(1):
				# fit
				# for set in fitset:
					# neuron.fit(set.X, 1, set.Y == Y)
				try:
					neuron.fit(list(map(NeuronM.Sample.to_num_class(Y), fitset)), max_iter=500)
					# check
					loger.log('fit ok', Y)
					success = len([1 for set in fitset if neuron.predict(set.X) == (set.Y == Y)])
				except NeuronM.Disconvergence:
					loger.log('split', 'disconv')
					success = -1
				if success == total:
					loger.log('split', 'succ == total')
					return neuron
				if success > last_good_rate:
					snapshot = neuron.save()
					last_good_rate = success
			if float(last_good_rate) / total > allow_accuracy:
				neuron.load(snapshot)
				loger.log('split', 'last good allow')
				return neuron
			else:
				loger.log('split','last good bad: %s of %s' % (float(last_good_rate) / total, allow_accuracy))
				return None

		def final_check(fit, check):
			percent = lambda l1, l2: int(100 * float(len(l1)) / len(l2))
			print('FIT:')
			f_fit = [(set.X,set.Y,self.predict(set.X)) for set in fit if self.predict(set.X) != set.Y]
			print('%s of %s (%s%% err)' % (len(f_fit), len(fit), percent(f_fit, fit)))
			f_che = [(set.X,set.Y,self.predict(set.X)) for set in check if self.predict(set.X) != set.Y]
			print('%s of %s (%s%% err)' % (len(f_che), len(check), percent(f_che, check)))
			return percent(f_fit, fit) < 10 and percent(f_che, check) < 10


		all_names = set(map(lambda a: a.Y, fitset))
		grouped = {}
		for name in all_names:
			grouped[name] = []
		for item in fitset:
			grouped[item.Y].append(item)

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
					fit = [s for s in fit if s.Y != neuron.name]
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
		print('NEURONS', self.neurons)
		self.neurons = []
		raise CantSplitError

def test():
	samp = lambda lst: NeuronM.Sample(lst[:-1], lst[-1])
	src = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'iris.csv')
	def make(s):
		try:
			return float(s)
		except:
			return s
	fit = [samp([make(t) for t in line.split(',')]) for line in open(src, 'rt').read().split('\n')]
	clsf = Classificator.from_fitset(fit)
	fst = clsf.neurons[0]
	if fst.name == 'Iris-setosa':
		print(fst.cost)

test()
'''
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
'''