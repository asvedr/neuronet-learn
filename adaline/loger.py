class Loger(object):
	"""docstring for Loger"""
	class _Loger(object):
		"""docstring for _Loger"""
		def __init__(self):
			super().__init__()
			self.header = open('log', 'wt')
		def log(self, label, text):
			if label:
				self.header.write(str(label) + ':')
			self.header.write(str(text))
			self.header.write('\n')
			self.header.flush()
		def close(self):
			self.header.close()
			self.__class__._instance = None	
	_instance = None
	def __init__(self):
		if Loger._instance is None:
			Loger._instance = Loger._Loger()
	def __getattr__(self, name):
		return getattr(Loger._instance, name)

def log(*args):
	Loger().log(*args)