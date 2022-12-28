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
def plot(indexes, policy, intervention):
    ax = plt.subplot(121)

    # plot vehicle speed change (state_transition index : -1=noint, 0=int)
    if intervention == -1:
        ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_no_int", alpha=0.5, c="green", linestyle="--")
    else:
        ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_int", alpha=0.5, c="orange")
    
    # plot intervention point 
    for risk_id in range(0, len(mdp.risk_positions)):
        int_indexes = [i for i, v in enumerate(policy) if v == risk_id]
        ax.scatter([mdp.index_value(indexes[i], 0)  for i in int_indexes], [mdp.index_value(indexes[i], 1)  for i in int_indexes], label="intervention", alpha=0.5, c=risk_colors[risk_id], linestyle="--")
        # plot risk position
        ax.axvspan(mdp.risk_positions[risk_id]-mdp.state_width[0], mdp.risk_positions[risk_id]+mdp.state_width[0], color=risk_colors[risk_id], alpha=0.2)

        # risk prob change
        if intervention == -1:
            print("intervention:", intervention, "speed:", [mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes])
            ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes], label="risk_prob_no_int", alpha=0.5, c=risk_colors[risk_id])
        else:
            print("intervention:", intervention, "speed", [mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes])
            ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes], label="risk_prob_int", alpha=0.5, c=risk_colors[risk_id])
         
    # plot no intervention point 
    int_indexes = [i for i, v in enumerate(policy) if v == -1]
    ax.scatter([mdp.index_value(indexes[i], 0)  for i in int_indexes], [mdp.index_value(indexes[i], 1)  for i in int_indexes], label="no_intervention", alpha=0.5, c="black")

    ax.legend()
    ax.set_ylim([0, 15])
    ax.set_xlabel("travel distance [m]")
    ax.set_ylabel("speed [m/s]")

def plot_performance():
    intercept_time = 3
    intercept_acc = 0.5
    slope = 0.25
    ax = plt.subplot(122)
    x_list = np.arange(0, 6)
    y = [None]*len(x_list)
    print(x_list)
    for i in range(len(x_list)):
        if x_list[int(i)] < intercept_time: continue
        y[int(i)] = (min(max(slope * (x_list[i] - intercept_time) + intercept_acc, 0.0), 1.0))

    ax.plot(x_list, y, label="acc")
    ax.set_xlim([0, x_list[-1]])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel("intervention time [s]")
    ax.set_ylabel("accuracy")



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
initial_state = [0, 11.2, -1, 0.5, 0.5]
egotistical_policy = [-1] * 100

param_list = ["param.yaml"]
intervention = 0 # 0:intervention -1:no_intervention
for param in param_list:
    print(param)
    with open(param) as f:
        param = yaml.safe_load(f)

    mdp = MDP(param)
    with open(param["filename"]+"_p.pkl", "rb") as f:
        p = pickle.load(f)

    index = tuple(mdp.to_index(initial_state))
    index_list = []
    policy_list = []

    while not mdp.final_state(index):
        policy = p[index]
        [prob, index_after] =  mdp.state_transition(policy, index)[intervention]
        index_list.append(index_after)
        policy_list.append(int(policy))
        index = tuple(index_after)

    print("policy", param, policy_list)
    plot(index_list, policy_list, intervention)

index_list, policy_list = myopic_policy(mdp, int_time=param["min_time"], initial_state=initial_state, intervention=intervention)
print(policy_list)
plot(index_list, policy_list, intervention)
plt.show()
