#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from ras_value_iteration_sweep_wo_eta import MDP

risk_colors = ["red", "blue", "green"]
intervention_color = ["green", "orange"]
mdp = MDP()
# mdp.init_state_space()
def plot(indexes, policy, intervention):
    ax_beav = plt.subplot(121)
    if intervention == 0:
        plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_int", alpha=0.5, c="green")
    else:
        plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], label="speed_no_int", alpha=0.5, c="orange")
    print("policy", policy)
    for int_state in range(0, len(mdp.risk_positions)):
        int_indexes = [i for i, v in enumerate(policy) if v == int_state]
        plt.scatter([mdp.index_value(indexes[i], 0)  for i in int_indexes], [mdp.index_value(indexes[i], 1)  for i in int_indexes], label="target", alpha=0.5, c=risk_colors[int_state])
        print(mdp.risk_state_index, int_state, indexes[0])
        plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+int_state) for i in indexes], label="risk_prob", alpha=0.5, c=risk_colors[int_state])
         
    # plt.plot([mdp.index_value(i, 0) for i in indexes], [mdp.get_int_performance(i) for i in indexes], label="int_acc", alpha=0.5)
    plt.legend()

def plot_performance(min_int_time, min_int_acc, acc_slope):
    ax_perf = plt.subplot(122)
    x_list = np.arange(mdp.operator_performance_min[0], mdp.operator_performance_max[0]+mdp.operator_performance_width[0], mdp.operator_performance_width[0])
    y = [None]*len(x_list)
    print(x_list)
    for i in range(len(x_list)):
        if x_list[int(i)] < min_int_time: continue
        y[int(i)] = (min(max(acc_slope * (x_list[i] - min_int_time) + min_int_acc, 0.0), 1.0))

    plt.plot(x_list, y, label="acc")
    plt.xlim([0, x_list[-1]])
    plt.ylim([0, 1.0])
with open("policy.pkl", "rb") as f:
    p = pickle.load(f)
# with open("value.pkl", "rb") as f:
#     v = pickle.load(f)
# p = [-1, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


# ego_pose, ego_speed, int_min_time, int_min_acc, int_slope, int_time, int_target, risk_prob, risk_eta
    

test_list = [
        [0, 50, 4, 0.5, 0.25, 0, -1, 0.25],
        [0, 50, 4, 0.5, 0.25, 0, -1, 0.5],
        [0, 50, 2, 0.5, 0.25, 0, -1, 0.5],
        [0, 50, 4, 0.5, 0.25, 0, -1, 1.0],
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
    plot_performance(test[2], test[3], test[4])
    plt.show()
