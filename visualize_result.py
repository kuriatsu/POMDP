#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from ras_value_iteration_sweep import MDP
import sys

risk_colors = ["red", "blue", "green"]
intervention_color = ["green", "orange"]
mdp = MDP()
# mdp.init_state_space()
def plot(indexes, policy, intervention):
    ax = plt.subplot(121)
    ax_right = ax.twinx()
    ax_right2 = ax.twinx()
    if intervention == -1:
        ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_no_int", alpha=0.5, c="green", linestyle="--")
    else:
        ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_int", alpha=0.5, c="orange")
    
    # plot intervention 
    for risk_id in range(0, len(mdp.risk_positions)):
        int_indexes = [i for i, v in enumerate(policy) if v == risk_id]
        ax.scatter([mdp.index_value(indexes[i], 0)  for i in int_indexes], [mdp.index_value(indexes[i], 1)  for i in int_indexes], label="intervention", alpha=0.5, c=risk_colors[risk_id], linestyle="--")
        ax.axvspan(mdp.risk_positions[risk_id]-mdp.state_width[0], mdp.risk_positions[risk_id]+mdp.state_width[0], color=risk_colors[risk_id], alpha=0.2)
        if intervention == -1:
            print(intervention, [mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes])
            ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes], label="risk_prob_no_int", alpha=0.5, c=risk_colors[risk_id])
        else:
            print(intervention, [mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes])
            ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+risk_id) for i in indexes], label="risk_prob_int", alpha=0.5, c=risk_colors[risk_id])
         
    # plot no intervention 
    int_indexes = [i for i, v in enumerate(policy) if v == -1]
    ax.scatter([mdp.index_value(indexes[i], 0)  for i in int_indexes], [mdp.index_value(indexes[i], 1)  for i in int_indexes], label="no_intervention", alpha=0.5, c="black")

    # plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.get_int_performance(i) for i in indexes], label="int_acc", alpha=0.5)
    ax.legend()
    ax_right.legend()
    ax_right.set_ylim([0, 1])
    ax_right2.legend()
    ax_right2.set_ylim([0, 1])
    ax_right.set_ylabel("crossing risk")
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

with open(sys.argv[1], "rb") as f:
    p = pickle.load(f)
# with open("value.pkl", "rb") as f:
#     v = pickle.load(f)
# p = [-1, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


# ego_pose, ego_speed, int_min_time, int_min_acc, int_slope, int_time, int_target, risk_prob, risk_eta
    

# test_list = [
#         [0, 50, 4, 0.5, 0.25, 0, -1, 0.25, 0.25],
#         [0, 50, 4, 0.5, 0.25, 0, -1, 0.5, 1.0],
#         [0, 50, 2, 0.5, 0.25, 0, -1, 0.5, 1.0],
#         [0, 50, 4, 0.5, 0.25, 0, -1, 1.0, 0.25],
#         ]
test_list = [
        [0, 50, 0, -1, 0.25, 1.0],
        [0, 50, 0, -1, 0.5, 0.5],
        [0, 50, 0, -1, 0.75, 0.25],
        [0, 50, 0, -1, 1.0, 0.5],
        ]
for test in test_list:
    for intervention in [0, -1]:
        human_intervention = True
        index = tuple(mdp.to_index(test))
        index_list = []
        policy_list = []
        i = 0
        while not mdp.final_state(index):
            # policy = p[i] 
            policy = p[index]
            [prob, index_after] =  mdp.state_transition(policy, index)[intervention]
            # print(mdp.action_value(policy, index)) # need to execute init_state_space
            index_list.append(index_after)
            policy_list.append(int(policy))
            index = tuple(index_after)
            i += 1

        plot(index_list, policy_list, intervention)
    plot_performance()
    print(test)
    plt.show()
