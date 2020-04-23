class Counter(DatasetEvaluator):
	"""
	This is a copy of the custom DatasetEvaluator example from detectron2
	It counts the number of output instances, but does not take the 
	confidence threshold into account
	"""
  	def reset(self):
    	self.count = 0
  	def process(self, inputs, outputs):
    	for output in outputs:
      		self.count += len(output["instances"])
  	def evaluate(self):
    	# save self.count somewhere, or print it, or return it.
    	return {"count": self.count}