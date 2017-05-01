from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy as sp

def print_docs(data, labels, filename):
	# create file
	f = open(filename, 'w+')

	# non-zero words for each document
	(rows, cols) = data.nonzero()

	str_doc = [0]
	prev_row = rows[0]	
	count = 0
	for row, col in zip(rows, cols):
		if prev_row != row: 
			# move to different document, output document string
			str_doc[0] = str(count)
			doc_str = " ".join(str_doc)
			f.write(doc_str)

			# reset parameters
			str_doc = [0]
			prev_row = row
			count = 0
		str_doc.append(":".join([str(col), str(data[row, col])]))
		count += 1

newsgroups = fetch_20newsgroups(subset='all')
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

vectorizer = CountVectorizer()
vectorizer.fit(newsgroups.data)

train_matrix = vectorizer.transform(train.data)
test_matrix = vectorizer.transform(test.data)

print_docs(train_matrix, train.target, "train.in")
print_docs(test_matrix, test.target, "test.in")
