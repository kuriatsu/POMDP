#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open("/home/kuriatsu/Dropbox/data/policy_400itr.pkl", "rb") as f:
    policy = pickle.load(f)

with open("/home/kuriatsu/Dropbox/data/value_400itr.pkl", "rb") as f:
    value = pickle.load(f)
print(len(policy[policy==0]))
v = value[:, :, 1, 4, 1, 2, 0, 2, 6]
sns.heatmap(np.rot90(v))
plt.ylim(0, v.shape[0])
plt.yticks(np.arange(0, 55.0/5.0, 2.0), np.arange(0, 55.0, 10))
plt.xticks(np.arange(0, 100.0/3.0, 10), np.arange(0, 100.0, 30))
plt.show()

p = policy[:, :, 1, 4, 1, 2, 0, 2, 6]
sns.heatmap(np.rot90(p), square=False)
plt.show()
