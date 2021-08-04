class UMLS_SIMILARITY:
	def __init__(self, umls1, umls2, score):
		self.umls1 = umls1
		self.umls2 = umls2
		self.score = score
		
	def get_umls1(self):
		return self.umls1

	def get_umls2(self):
		return self.umls2

	def get_score(self):
		return self.score




	def __str__():
 		return "umls1: " + umls1 + " , " + "umls2: " + umls2 + " , " + "score: " + score
