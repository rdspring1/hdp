import numpy as np
from numpy import random
import string
import random

def generate(N, L, M, D, K, theta):
	"""
		N = Number of Documents
		L = Number of Words
		M = Number of Topics
		D = Words per Document
		K = Topics per Document
		theta = topic multinomial distributions
	Generate data:
		1) Sample the number of entities K using log-normal prior
		2) For each entity, draw a distribution of tokens from a Dirichlet prior.
		3) For each mention, assign an entity using uniform distribution
		4) Then, using the mention's entity, draw a token from the entity's dictionary.
	"""
	data = [] 
	for idx in range(N):
		doc = dict()
		topics = np.random.choice(M, K)
		for word in range(D):
			topic = random.choice(topics)
			token = np.random.choice(L, p=theta[:, topic])
			if token not in doc:
				doc[token] = 1
			else:
				doc[token] += 1
		data.append(doc)
	return data

def print_docs(data):
	for doc in data:
		str_doc = []
		M = len(doc)
		str_doc.append(str(M))
		for term, count in doc.items():
			str_doc.append(":".join([str(term), str(count)]))
		doc_str = " ".join(str_doc)
		print(doc_str)

N = 100
L = 12
M = 5
D = 50
K = 2

theta = np.ones((L, M))
theta[0, 0] = 2
theta[1:8, 0] = 10

theta[0, 1] = 10 
theta[1, 1] = 2
theta[2:8, 1] = 10

theta[9, 2] = 10
theta[10, 2] = 2
theta[11, 2] = 2

theta[9, 3] = 2
theta[10, 3] = 10 
theta[11, 3] = 2

theta[9, 4] = 2
theta[10, 4] = 2
theta[11, 4] = 10

theta_p = theta / np.sum(theta, axis=0)
print_docs(generate(N, L, M, D, K, theta_p))
