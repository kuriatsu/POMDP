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
trajectory_color = "green"

# mdp.init_state_space()
def plot(mdp, ax, indexes, policyes, intervention, risk_num, cumlative_risk, travel_time):
    # plot vehicle speed change (state_transition index : -1=noint, 0=int)
    ax_risk = ax.twinx()
    ax_risk.set_ylim((0.0, 1.0))

    # plot risk position
    for risk_id in range(0, len(mdp.risk_positions)):
        ax.axvspan(mdp.risk_positions[risk_id]-mdp.state_width[0], mdp.risk_positions[risk_id]+mdp.state_width[0], color=risk_colors[risk_id], alpha=0.2)

    # plot trajectory
    ax.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, 1) for i in indexes], alpha=0.5, c=trajectory_color, label="speed")

    # plot risk prob
    for r in range(risk_num):
        ax_risk.plot([mdp.index_value(i, 0) for i in indexes], [mdp.index_value(i, mdp.risk_state_index+r) for i in indexes], alpha=0.5, c=risk_colors[r], label="risk prob")

    # plot intervention request
    for i, p in enumerate(policyes):
        if p != -1:
            ax_risk.plot(mdp.index_value(indexes[i], 0), 0.5, c=risk_colors[p], marker="x", label="int request")
        else:
            ax_risk.plot(mdp.index_value(indexes[i], 0), 0.5, c="grey", marker="x")


    ax.annotate("travel time: "+str(travel_time), xy=(10, 5), size=10)
    ax.annotate("cumulative ambiguity: "+str(cumlative_risk), xy=(10, 3), size=10, color="red")
    ax.set_xlabel("distance [m]", fontsize=14) 
    ax.set_ylabel("speed [m/s]", fontsize=14) 
    ax_risk.set_ylabel("risk probability", fontsize=14) 

def main():
    initial_state = [0, 14, 0, -1, 0.75, 0.5]
    intervention_list = [-1, 0] # -1:no intervention 0:intervention
    egotistical_policy = [-1] * 100

    param_list = [
            "param_19.yaml", 
            "param_20.yaml", 
            # "param_3.yaml",
            # "param_3.yaml",
            # "param_6.yaml",
            # "param_7.yaml",
            # "param_8.yaml",
            # "param_9.yaml",
            # "param_10.yaml",
            # "param_11.yaml",
            # "param_12.yaml",
            ]
    fig, axes = plt.subplots(len(param_list)+1, 1, sharex="all", tight_layout=True)

    index_list = []
    policy_list = []
    for idx, param_file in enumerate(param_list):
        print(param_file)
        with open(param_file) as f:
            param = yaml.safe_load(f)

        mdp = MDP(param)
        filename = param_file.split("/")[-1].split(".")[0]
        with open(f"{filename}_p.pkl", "rb") as f:
            p = pickle.load(f)

        index = tuple(mdp.to_index(initial_state))
        indexes = []
        policyes = []
        cumlative_risk = 0
        travel_time = 0
        while not mdp.final_state(index):
            policy = p[index]

            intervention = 0 if mdp.index_value(index, mdp.int_state_index+1) == -1 else intervention_list[int(mdp.index_value(index, mdp.int_state_index+1))]
            index_after_list =  mdp.state_transition(policy, index)
            max_p = max([i[0] for i in index_after_list])
            highest_index_list = [i for i, x in enumerate(index_after_list) if x[0]==max_p]
            index_after = index_after_list[highest_index_list[intervention]][1]
            
            for i, risk_position in enumerate(mdp.risk_positions):
                pos = mdp.index_value(index, 0)
                pos_after = mdp.index_value(index_after, 0)
                speed = mdp.index_value(index, 1)
                if pos <= risk_position < pos_after and speed > mdp.min_speed:
                    cumlative_risk += (0.5 - abs(mdp.index_value(index_after, mdp.risk_state_index+i) - 0.5))*2

            indexes.append(index_after)
            policyes.append(int(policy))
            index = tuple(index_after)

        travel_time = len(indexes)*1
        plot(mdp, axes[idx], indexes, policyes, 0, len(mdp.risk_positions), cumlative_risk, travel_time)
        
    with open(param_list[1]) as f:
        param = yaml.safe_load(f)
    mdp = MDP(param)
    indexes, policyes, cumlative_risk, travel_time = myopic_policy(mdp, 5, initial_state, intervention_list)
    plot(mdp, axes[-1], indexes, policyes, 0, len(mdp.risk_positions), cumlative_risk, travel_time)
    index_list.append(indexes)
    policy_list.append(policyes)

        # plot(mdp, axes, [index_list], [policy_list], intervention, 2)
    plt.show()

if __name__ == "__main__":
    main()
