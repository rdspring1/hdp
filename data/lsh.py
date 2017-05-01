import numpy as np
from numpy import random
import scipy as sp
import scipy.stats
import string
import random
import math

eta = 0.5
gamma = 0.1
alpha = 1.0

# Number of Documents
N = 100

# Words per Document
D = 50

# Topics per Document
K = 2

# Number of Words
L = 12

# Number of Topics
M = 5

# Topic / Word Distribution
theta = np.ones((L, M))
theta[0, 0] = 1
theta[1:8, 0] = 10

theta[0, 1] = 10 
theta[1, 1] = 1
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

'''
N = 100
K = L
theta = np.random.choice(L, (N, K))
def wta(x, y, theta):
    match = 0 
    for perm in theta:
        xmax = np.argmax(x[perm]) 
        ymax = np.argmax(y[perm]) 
        if xmax == ymax:
            match += 1
    return match
'''

N = 1000
K = L
Width = 2
theta = np.random.choice(L, (N, K))

def first_nonzero(data):
    for idx in range(len(data)):
        if data[idx] > 0:
            return idx
    return len(data)

def minhash(x, y, theta):
    match = 1 
    count = 0
    for perm in theta:
        xidx = first_nonzero(x[perm]) 
        yidx = first_nonzero(y[perm]) 
        if xidx == yidx:
            count += 1
        else:
            count = 0
		
        if count == Width:
            match += 1
            count = 0
    return match

N = 1000
K = 6
Width = 5 
theta = np.random.choice(L, (N, K))
def wta(x, y, theta):
    match = 1 
    count = 0
    for perm in theta:
        xmax = np.argmax(x[perm]) 
        ymax = np.argmax(y[perm]) 
        if xmax == ymax:
            count += 1
        else:
            count = 0
		
        if count == Width:
            match += 1
            count = 0
    return match
             
def pairwise_order(x, y, L):
    count = 0
    #total = L*(L-1)/2.0
    for idx in range(1, L, 1):
        for jdx in range(0, idx, 1):
            left = x[idx] - x[jdx]
            right = y[idx] - y[jdx]
            lT = 1 if left >= 0 else -1
            rT = 1 if right >= 0 else -1
            count += lT * rT
    return count

def cosine_sim(x, y):
	cosine = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
	return 1.0 - np.arccos(cosine) / math.pi

def jaccard_sim(x, y, L):
    intersect = 0
    total = 0
    for idx in range(L):
        intersect += min(x[idx], y[idx])
        total += max(x[idx], y[idx])
    return intersect / total 

# A set of tables belong to a topic
# Each table represents a set of words from the topic
# 80 tables in the set
# 25 words per table

T = 100
WPT = 100

first = 0
second = 1

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
gap = 0
correct = 0
for idx in range(T):
    item = data[idx, :]
    topic = topic_idx[idx]

    #tau1, left_sim = sp.stats.kendalltau(left, item)
    #tau2, right_sim = sp.stats.kendalltau(right, item)

    #left_sim = pairwise_order(left, item, L)
    #right_sim = pairwise_order(right, item, L)

    #left_sim = minhash(left, item, theta)
    #right_sim = minhash(right, item, theta)

    left_sim = jaccard_sim(left, item, L)
    right_sim = jaccard_sim(right, item, L)

    left_p = left_sim / (left_sim + right_sim)
    right_p = 1.0 - left_p 
    gap += abs(left_p - right_p)
    if random.random() <= left_p:
        if topic == first:
            correct += 1
        left_set.append(item)
    else:
        if topic == second:
            correct += 1
        right_set.append(item)

print(correct)
print(gap/T)
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
