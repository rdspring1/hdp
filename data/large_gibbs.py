import numpy as np
from numpy import random
import scipy as sp
import scipy.stats
from scipy.special import gammaln
import string
import random
import math
import time

# Number of Words
L = 10000

# Number of Topics
M = 5

# Topic / Word Distribution
theta = np.zeros((L, M))

for idx in range(M):
	num_peaks = 1000
	peaks = np.random.choice(L, (num_peaks))
	theta[peaks, idx] = 1.0
theta_p = (theta / np.sum(theta, axis=0))

def gibbs(num_tables, topic, table, eta=0.5):
    # initially 1 table for each topic
	num_tables_per_topic = np.log(num_tables)
	v_eta = L * eta
	num_words_per_table = np.sum(table)
	num_words_per_topic = np.sum(topic)
	left = gammaln(v_eta + num_words_per_topic) - gammaln(v_eta + num_words_per_topic + num_words_per_table)
	right = 0
	for idx in range(L):
		num_words_per_topic_per_word = topic[idx]
		num_words_per_table_per_word = table[idx]
		right += gammaln(eta + num_words_per_topic_per_word + num_words_per_table_per_word) - gammaln(eta + num_words_per_topic_per_word)
	#return num_tables_per_topic + left + right
	return right

def log_sum(log_a, log_b):
    if log_a < log_b:
        return log_b + np.log(1 + np.exp(log_a-log_b))
    else:
        return log_a + np.log(1 + np.exp(log_b-log_a))

# A set of tables belong to a topic
# Each table represents a set of words from the topic
first = 0
second = 3

T = 100
WPT = 1000
data = np.zeros((T, L))
topic_idx = np.zeros((T))
for idx in range(T):
	topic = np.random.choice([first, second])
	topic_idx[idx] = topic
	for jdx in range(WPT):
		token = np.random.choice(L, p=theta_p[:, topic])
		data[idx, token] += 1

# new cluster centers
left = np.zeros((L,))
right = np.zeros((L,))
for idx in range(WPT):
	left_token = np.random.choice(L, p=theta_p[:, first])
	right_token = np.random.choice(L, p=theta_p[:, second])
	left[left_token] += 1
	right[right_token] += 1
print(left)
print(right)

# estimate
print("estimate")
left_set = [left]
right_set = [right]
correct = 0

start_time = time.time()
for idx in range(T):
    item = data[idx, :]
    topic = topic_idx[idx]

    left_sim = max(gibbs(len(left_set), np.sum(left_set, axis=0), item), 1e-10)
    right_sim = max(gibbs(len(right_set), np.sum(right_set, axis=0), item), 1e-10)

    left_p = left_sim - log_sum(left_sim, right_sim)
    if np.log(random.random()) <= left_p:
        if topic == first:
            correct += 1
        left_set.append(item)
    else:
        if topic == second:
            correct += 1
        right_set.append(item)

end_time = time.time()
print(end_time - start_time)
print(correct)

left_estimate = np.mean(left_set, axis=0)
right_estimate = np.mean(right_set, axis=0)
print(left_estimate)
print(right_estimate)

# correct
print("correct")
correct_left_set = []
correct_right_set = []
for idx in range(T):
    item = data[idx, :]
    if topic_idx[idx]:
        correct_left_set.append(item)
    else:
        correct_right_set.append(item)

left_correct = np.mean(correct_left_set, axis=0)
right_correct = np.mean(correct_right_set, axis=0)
print(left_correct / np.sum(left_correct))
print(right_correct / np.sum(right_correct))

print("probabilty")
print(theta_p[:, 0])
print(theta_p[:, 1])
