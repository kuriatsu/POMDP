#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from ras_value_iteration_sweep import MDP
import sys

# mdp = MDP()
# with open("/home/kuriatsu/Dropbox/data/policy_400itr.pkl", "rb") as f:
with open(sys.argv[1], "rb") as f:
    policy = pickle.load(f)

# ego pose, ego_speed, operator min_tim, operator_min_acc, operator_slope, int time, int_target,  target_prob
# with open("/home/kuriatsu/Dropbox/data/value_400itr.pkl", "rb") as f:
with open(sys.argv[2], "rb") as f:
    value = pickle.load(f)
# v = value[:, :, 1, 1, 0, 2, 0, 4, 4]
# index = [0, 0, 1, 1, 1, 2, 0, 4, 4]
v = value[:, :, 0, 0, 0, 0]
p = policy[:, :, 0, 0, 0, 0]
# print((index + mdp.state_min) * mdp.state_width)
# plt.yticks(np.arange(0, 55.0/5.0, 2.0), np.arange(0, 55.0, 10))
# plt.xticks(np.arange(0, 100.0/3.0, 10), np.arange(0, 100.0, 30))

sns.heatmap(np.rot90(p, -1), square=False)
plt.show()

sns.heatmap(np.rot90(v, -1), square=False)
plt.ylim(0, v.shape[0])
plt.show()
