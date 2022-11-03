#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from ras_value_iteration_sweep import MDP


mdp = MDP()
def plot(indexes, policy, plt):
    plt.plot([mdp.index_value(i, 0) for i in indexes], [i[1] for i in indexes], label="speed")

    plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 6)  for i in indexes], label="target")
    plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 7) for i in indexes], label="risk_prob")
    plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.get_int_performance(i) for i in indexes], label="int_acc")
    plt.legend()
    plt.show()

with open("policy.pkl", "rb") as f:
    p = pickle.load(f)

with open("value.pkl", "rb") as f:
    v = pickle.load(f)

fig, ax = plt.subplots()

# ego_pose, ego_speed, int_min_time, int_min_acc, int_slope, int_time, int_target, risk_prob, risk_eta
    
test_list = [
        [0, 50, 2, 0.5, 0.25, 0, -1, 0.25, 4],
        ]
for test in test_list:
    human_intervention = True
    index = tuple(mdp.to_index(test))
    index_list = []
    policy_list = []
    while not mdp.final_state(index):
        policy = p[index]
        [prob, index_after] =  mdp.state_transition(policy, index)[0]
        index_list.append(index_after)
        policy_list.append(policy)
        index = tuple(index_after)
    
    print(index_list, policy_list)
    plot(index_list, policy, plt)
