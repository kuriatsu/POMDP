#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import yaml
from ras_value_iteration_sweep import MDP
from myopic_agent import myopic_policy
import sys

risk_colors = ["red", "blue", "green"]
intervention_color = ["green", "orange"]

# mdp.init_state_space()
def plot(index_list, policy_list, intervention, risk_num):
    fix, axes = plt.subplots(1, 1+len(policy_list), sharex="all", tight_layout=True)
    print(index_list, policy_list)
    # plot vehicle speed change (state_transition index : -1=noint, 0=int)
    for idx, indexes in enumerate(index_list):
        ax_traj = axes[0]
        ax_risk = axes[idx]
        if intervention == -1:
            ax_traj.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], alpha=0.5, c="green", linestyle="--")
            for r in range(risk_num):
                ax_risk.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+r) for i in indexes], alpha=0.5, c=risk_colors[r], linestyle="--")
        else:
            ax_traj.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], alpha=0.5, c="orange")
            for r in range(risk_num):
                ax_risk.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+r) for i in indexes], alpha=0.5, c=risk_colors[r])
    
        for policy in policy_list[idx]:
            for p in policy:
                if p != -1:
                    ax_risk.plot(mdp.index_value(idx, 0), 0.5, c=risk_colors[p])


# ego_pose, ego_speed, int_time, int_target, risk_prob
# test_list = [
#         [0, 50, 4, 0.5, 0.25, 0, -1, 0.25, 0.25],
#         [0, 50, 4, 0.5, 0.25, 0, -1, 0.5, 1.0],
#         [0, 50, 2, 0.5, 0.25, 0, -1, 0.5, 1.0],
#         [0, 50, 4, 0.5, 0.25, 0, -1, 1.0, 0.25],
#         ]
# test_list = [
#         [0, 50, 0, -1, 0.25, 1.0],
#         [0, 50, 0, -1, 0.5, 0.5],
#         [0, 50, 0, -1, 0.75, 0.25],
#         [0, 50, 0, -1, 1.0, 0.5],
#         ]
initial_state = [0, 14, 0, -1, 0.5, 0.5]
egotistical_policy = [-1] * 100

param_list = ["param.yaml"]
intervention = 0 # 0:intervention -1:no_intervention
index_list = []
policy_list = []

for param in param_list:
    print(param)
    with open(param) as f:
        param = yaml.safe_load(f)

    mdp = MDP(param)
    with open(param["filename"]+"_p.pkl", "rb") as f:
        p = pickle.load(f)

    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policyes = []

    while not mdp.final_state(index):
        policy = p[index]
        index_after_list =  mdp.state_transition(policy, index)
        index_after = None
        max_p = 0.0
        high_index = None
        for i, v in enumerate(index_after_list):
            print(max_p, high_index, v[0])
            if max_p < v[0]:
                max_p = v[0]
                index_after = v[1]
                high_index = i

        print("result", max_p, index_after, high_index)
        indexes.append(index_after)
        policyes.append(int(policy))
        index = tuple(index_after)

    index_list.append(indexes)
    policy_list.append(policyes)
    
print("policy", param, policy_list)
plot(index_list, policy_list, intervention, 2)

index_list, policy_list = myopic_policy(mdp, int_time=param["min_time"], initial_state=initial_state, intervention=intervention)
print(policy_list)
plot(index_list, policy_list, intervention)
plt.show()
