#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import pickle
import yaml
from ras_value_iteration_sweep import MDP
from myopic_agent import myopic_policy
import sys
import random


risk_colors = ["red", "blue", "green"]
trajectory_color = "green"

# mdp.init_state_space()
def plot(mdp, ax, indexes, policies, intervention, risk_num, cumlative_risk, travel_time, title):
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
    for i, p in enumerate(policies):
        if p != -1:
            ax_risk.plot(mdp.index_value(indexes[i], 0), 0.5, c=risk_colors[p], marker="x", label="int request")
        else:
            ax_risk.plot(mdp.index_value(indexes[i], 0), 0.5, c="grey", marker="x")


    ax.annotate("travel time: "+str(travel_time), xy=(10, 5), size=10)
    ax.annotate("cumulative ambiguity: "+str(cumlative_risk), xy=(10, 3), size=10, color="red")
    ax.set_xlabel("distance [m]", fontsize=14) 
    ax.set_ylabel("speed [m/s]", fontsize=14) 
    ax.set_title(title, fontsize=14, y=-0.25)
    ax_risk.set_ylabel("risk probability", fontsize=14) 

def pomdp_agent(mdp, policy, initial_state, intervention_list):
    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policies = []
    cumlative_risk = 0
    while not mdp.final_state(index):
        p = policy[index]

        intervention = 0 if mdp.index_value(index, mdp.int_state_index+1) == -1 else intervention_list[int(mdp.index_value(index, mdp.int_state_index+1))]
        index_after_list =  mdp.state_transition(p, index)
        max_p = max([i[0] for i in index_after_list])
        highest_index_list = [i for i, x in enumerate(index_after_list) if x[0]==max_p]
        print(index_after_list)
        index_after = index_after_list[highest_index_list[intervention]][1]
        
        for i, risk_position in enumerate(mdp.risk_positions):
            pos = mdp.index_value(index, 0)
            pos_after = mdp.index_value(index_after, 0)
            speed = mdp.index_value(index, 1)
            if pos <= risk_position < pos_after and speed > mdp.min_speed:
                cumlative_risk += (0.5 - abs(mdp.index_value(index_after, mdp.risk_state_index+i) - 0.5))*2

        indexes.append(index_after)
        policies.append(int(p))
        index = tuple(index_after)

    travel_time = len(indexes)
    request_time = len([p for p in policies if p!=-1])
    return indexes, policies, cumlative_risk, travel_time, request_time
    
def egotistical_agent(mdp, initial_state, intervention_list):
    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policies = []
    cumlative_risk = 0
    while not mdp.final_state(index):
        policy = -1 

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
        policies.append(int(policy))
        index = tuple(index_after)

    travel_time = len(indexes)
    request_time = len([p for p in policies if p!=-1])
    return indexes, policies, cumlative_risk, travel_time, request_time

def main():

    param_list = [
            # "param_2_1.yaml",
            # "param_2_2.yaml",
            # "param_2_3.yaml",
            # "param_2_4.yaml",
            # "param_2_5_1.yaml",
            # "param_2_5_2.yaml",
            # "param_2_5_3.yaml",
            # "param_2_5_4.yaml",
            # "param_2_6.yaml",
            # "param_2_7.yaml",
            # "param_2_8.yaml",
            # "param_2_9.yaml",
            # "param_2_10.yaml",
            # "param_2_11.yaml",
            # "param_2_12.yaml",
            # "param_2_13.yaml",
            # "param_2_14.yaml",
            # "param_2_15.yaml",
            # "param_3_1.yaml",
            # "param_3_2.yaml",
            # "param_3_3.yaml",
            # "param_3_4.yaml",
            # "param_3_6.yaml",
            # "param_3_7.yaml",
            # "param_3_8.yaml",
            # "param_3_9.yaml",
            # "param_3_10.yaml",
            # "param_3_11.yaml",
            # "param_3_12.yaml",
            # "param_3_13.yaml",
            # "param_3_14.yaml",
            # "param_3_15.yaml",
            # "param_4_1.yaml",
            # "param_4_2.yaml",
            # "param_4_4.yaml",
            # "param_4_4.yaml",
            # "param_4_6.yaml",
            # "param_4_7.yaml",
            # "param_4_8.yaml",
            # "param_4_9.yaml",
            # "param_4_10.yaml",
            # "param_4_11.yaml",
            # "param_4_12.yaml",
            # "param_4_13.yaml",
            # "param_4_14.yaml",
            # "param_4_15.yaml",
            # "param_5_1.yaml",
            # "param_5_2.yaml",
            # "param_5_5.yaml",
            # "param_5_4.yaml",
            # "param_5_6.yaml",
            # "param_5_7.yaml",
            # "param_5_8.yaml",
            # "param_5_9.yaml",
            # "param_5_10.yaml",
            # "param_5_11.yaml",
            # "param_5_12.yaml",
            # "param_5_13.yaml",
            # "param_5_14.yaml",
            # "param_5_15.yaml",
            # "param_6_1.yaml",
            # "param_6_2.yaml",
            # "param_6_6.yaml",
            # "param_6_4.yaml",
            # "param_6_6.yaml",
            # "param_6_7.yaml",
            # "param_6_8.yaml",
            # "param_6_9.yaml",
            # "param_6_10.yaml",
            # "param_6_11.yaml",
            # "param_6_12.yaml",
            # "param_6_13.yaml",
            # "param_6_14.yaml",
            # "param_6_15.yaml",
            # "param_7_1.yaml",
            # "param_7_2.yaml",
            # "param_7_7.yaml",
            # "param_7_4.yaml",
            # "param_7_6.yaml",
            # "param_7_7.yaml",
            # "param_7_8.yaml",
            # "param_7_9.yaml",
            # "param_7_10.yaml",
            # "param_7_11.yaml",
            # "param_7_12.yaml",
            # "param_7_13.yaml",
            # "param_7_14.yaml",
            # "param_7_15.yaml",
            # "param_8_1.yaml",
            # "param_8_2.yaml",
            # "param_8_8.yaml",
            # "param_8_4.yaml",
            # "param_8_6.yaml",
            # "param_8_7.yaml",
            # "param_8_8.yaml",
            # "param_8_9.yaml",
            # "param_8_10.yaml",
            # "param_8_11.yaml",
            # "param_8_12.yaml",
            # "param_8_13.yaml",
            # "param_8_14.yaml",
            # "param_8_15.yaml",
            "param_13_1.yaml",
            "param_13_2.yaml",
            "param_13_3.yaml",
            "param_13_4.yaml",
            "param_13_5.yaml",
            "param_13_6.yaml",
            "param_13_7.yaml",
            "param_13_8.yaml",
            "param_13_9.yaml",
            # "param_13_10.yaml",
            "param_13_11.yaml",
            ]
    # fig, axes = plt.subplots(len(param_list), 3, sharex="all", tight_layout=True)
    # fig_eval, ax_eval = plt.subplots(1, 3, tight_layout=True)

    initial_states = [
            [0, 11.2, 0, -1, 0.0, 0.0],
            [0, 11.2, 0, -1, 0.25, 0.0],
            [0, 11.2, 0, -1, 0.5, 0.0],
            [0, 11.2, 0, -1, 0.75, 0.0],
            [0, 11.2, 0, -1, 1.0, 0.0],
            [0, 11.2, 0, -1, 0.0, 0.25],
            [0, 11.2, 0, -1, 0.25, 0.25],
            [0, 11.2, 0, -1, 0.5, 0.25],
            [0, 11.2, 0, -1, 0.75, 0.25],
            [0, 11.2, 0, -1, 1.0, 0.25],
            [0, 11.2, 0, -1, 0.0, 0.5],
            [0, 11.2, 0, -1, 0.25, 0.5],
            [0, 11.2, 0, -1, 0.5, 0.5],
            [0, 11.2, 0, -1, 0.75, 0.5],
            [0, 11.2, 0, -1, 1.0, 0.5],
            [0, 11.2, 0, -1, 0.0, 0.75],
            [0, 11.2, 0, -1, 0.25, 0.75],
            [0, 11.2, 0, -1, 0.5, 0.75],
            [0, 11.2, 0, -1, 0.75, 0.75],
            [0, 11.2, 0, -1, 1.0, 0.75],
            [0, 11.2, 0, -1, 0.0, 1.0],
            [0, 11.2, 0, -1, 0.25, 1.0],
            [0, 11.2, 0, -1, 0.5, 1.0],
            [0, 11.2, 0, -1, 0.75, 1.0],
            [0, 11.2, 0, -1, 1.0, 1.0],
            ]

    # -1:no intervention 0:intervention
    intervention_lists = [
            [-1, -1], 
            [-1, 0], 
            [0, -1], 
            [0, 0], 
            ]
    result_list = pd.DataFrame(columns=["param", "risk", "int", "agent", "cumlative_risk", "travel_time", "request_time"])
    for initial_state in initial_states:
        for intervention_list in intervention_lists:
            for idx, param_file in enumerate(param_list):
                with open(param_file) as f:
                    param = yaml.safe_load(f)

                mdp = MDP(param)
                filename = param_file.split("/")[-1].split(".")[0]
                # with open(f"/run/media/kuriatsu/KuriBuffaloPSM/pomdp_intervention_target/experiment/{filename}_p.pkl", "rb") as f:
                with open(f"{filename}_p.pkl", "rb") as f:
                    p = pickle.load(f)

                buf_list = pd.DataFrame(columns=result_list.columns)

                indexes, policies, cumlative_risk, travel_time, request_time = pomdp_agent(mdp, p, initial_state, intervention_list)
                buf = pd.DataFrame([[filename, str(initial_state[-2:]), str(intervention_list), "pomdp", cumlative_risk, travel_time, request_time]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)
                # print(filename, buf_list)
                # plot(mdp, axes[idx, 0], indexes, policies, 0, len(mdp.risk_positions), cumlative_risk, travel_time, filename)

                indexes, policies, cumlative_risk, travel_time, request_time = myopic_policy(mdp, 5, initial_state, intervention_list)
                buf = pd.DataFrame([[filename, str(initial_state[-2:]), str(intervention_list), "myopic", cumlative_risk, travel_time, request_time]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)
                # print("myopic", buf_list)
                # plot(mdp, axes[idx, 1], indexes, policies, 0, len(mdp.risk_positions), cumlative_risk, travel_time, "myopic")

                indexes, policies, cumlative_risk, travel_time, request_time = egotistical_agent(mdp, initial_state, intervention_list)
                buf = pd.DataFrame([[filename, str(initial_state[-2:]), str(intervention_list), "egostistical", cumlative_risk, travel_time, request_time]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)
                # print("egostistical", buf_list)
                # plot(mdp, axes[idx, 2], indexes, policies, 0, len(mdp.risk_positions), cumlative_risk, travel_time, "egostistical")

                result_list = pd.concat([result_list, buf_list], ignore_index=True)
                # sns.lineplot(buf_list, x="agent", y="cumlative_risk", ax=ax_eval[0], label=str(initial_state[-2:])+str(intervention_list)) 
                # sns.lineplot(buf_list, x="agent", y="travel_time", ax=ax_eval[1], label=str(initial_state[-2:])+str(intervention_list)) 
                # sns.lineplot(buf_list, x="agent", y="request_time", ax=ax_eval[2], label=str(initial_state[-2:])+str(intervention_list)) 

    # plt.show()
    result_list.to_csv("result_13.csv")
    # fig_eval.savefig("result.svg")

def main_2():
    df = pd.read_csv(sys.argv[1])
    param_list = ["param_13_1",
                "param_13_2",
                "param_13_3",
                "param_13_4",
                "param_13_5",
                "param_13_6",
                "param_13_7",
                "param_13_8",
                "param_13_9",
                "param_13_10",
                "param_13_11",
                "param_13_12",
                "param_13_13",
                "param_13_14",
                "param_13_15",
                ]
    scenario_list = [
            "risk:[0.25, 0.25],int:[0, 0]",
            "risk:[0.5, 0.25],int:[0, 0]",
            "risk:[0.75, 0.25],int:[0, 0]",
            "risk:[0.25, 0.25],int:[-1, -1]",
            "risk:[0.5, 0.25],int:[-1, -1]",
            "risk:[0.75, 0.25],int:[-1, -1]",
            "risk:[0.25, 0.5],int:[0, 0]",
            "risk:[0.5, 0.5],int:[0, 0]",
            "risk:[0.75, 0.5],int:[0, 0]",
            "risk:[0.25, 0.5],int:[-1, -1]",
            "risk:[0.5, 0.5],int:[-1, -1]",
            "risk:[0.75, 0.5],int:[-1, -1]",
            "risk:[0.25, 0.75],int:[0, 0]",
            "risk:[0.5, 0.75],int:[0, 0]",
            "risk:[0.75, 0.75],int:[0, 0]",
            "risk:[0.25, 0.75],int:[-1, -1]",
            "risk:[0.5, 0.75],int:[-1, -1]",
            "risk:[0.75, 0.75],int:[-1, -1]",
            ]

    # target_scenario_df = df[(df["param"].isin(param_list)) & (df["scenario"].isin(scenario_list))]
    target_scenario_df = df[(df["param"].isin(param_list))]

    result_df = pd.DataFrame(columns=["agent", "intervention", "risk-ave", "risk-var", "risk-count", "trav-ave", "trav-var", "req-ave", "req-var"])
    for agent in target_scenario_df.agent.drop_duplicates():
        target_df = target_scenario_df[(target_scenario_df.agent == agent)]
        # initial_risk = ",".join(str(target_df.scenario).split(",")[:2])
        # operator_intervention = ",".join(str(target_df.scenario).split(",")[2:])
        if target_df.intervention.str in ["[-1, 0]", "[0, -1]"]:
            continue
        target_int_df = target_df[target_df.intervention == ("[0, 0]")]
        buf = pd.DataFrame([[
            agent, 
            "intervention",
            target_int_df.cumlative_risk.mean(),
            target_int_df.cumlative_risk.std(),
            len(target_int_df[target_int_df.cumlative_risk>0.0]),
            target_int_df.travel_time.mean(),
            target_int_df.travel_time.std(),
            target_int_df.request_time.mean(),
            target_int_df.request_time.std(),
            ]], columns=result_df.columns)
        result_df = pd.concat([result_df, buf], ignore_index=True)

        target_noint_df = target_df[target_df.intervention == ("[-1, -1]")]
        # target_noint_df = target_df[target_df.scenario.str.endswith("int:[-1, -1]")]
        buf = pd.DataFrame([[
            agent, 
            "no",
            target_noint_df.cumlative_risk.mean(),
            target_noint_df.cumlative_risk.std(),
            len(target_noint_df[target_noint_df.cumlative_risk>0.0]),
            target_noint_df.travel_time.mean(),
            target_noint_df.travel_time.std(),
            target_noint_df.request_time.mean(),
            target_noint_df.request_time.std(),
            ]], columns=result_df.columns)
        result_df = pd.concat([result_df, buf], ignore_index=True)

    result_df.to_csv("summary_13.csv")

    result_df = pd.DataFrame(columns=["param", "initial_risk", "operator_intervention", "cumlative_risk", "travel_time", "request_time"])
    for param in target_scenario_df.param.drop_duplicates():
        for intervention in target_scenario_df.intervention.drop_duplicates():
            for risk in target_scenario_df.risk.drop_duplicates():
                target_df = target_scenario_df[(target_scenario_df.risk == risk) & (target_scenario_df.intervention == intervention) & (target_scenario_df.param == param)]
                # initial_risk = ",".join(scenario.split(",")[:2])
                # operator_intervention = ",".join(scenario.split(",")[2:])
                if target_df.intervention.iloc[-1] in ["[-1, 0]", "[0, -1]"]:
                    continue
                cumlative_risk = target_df[target_df.agent=="pomdp"].cumlative_risk.iloc[-1] - target_df[target_df.agent=="myopic"].cumlative_risk.iloc[-1]
                travel_time = target_df[target_df.agent=="pomdp"].travel_time.iloc[-1] - target_df[target_df.agent=="myopic"].travel_time.iloc[-1]
                request_time = target_df[target_df.agent=="pomdp"].request_time.iloc[-1] - target_df[target_df.agent=="myopic"].request_time.iloc[-1]
                buf = pd.DataFrame([[param, risk, intervention, cumlative_risk, travel_time, request_time]], columns=result_df.columns)
                print(target_df)
                result_df = pd.concat([result_df, buf], ignore_index=True)
    result_df.to_csv("comparison_13.csv")
    print("increase risk", result_df[result_df.cumlative_risk>0.0])
    print("degrede efficiency", result_df[(result_df.travel_time>0.0) & (result_df.request_time>0.0)])

    # fig, axes = plt.subplots()
    sns.scatterplot(data=result_df, x="travel_time", y="request_time", hue="initial_risk", style="operator_intervention", s=50.0, alpha=1.0)
    # sns.lmplot(data=result_df, x="travel_time", y="request_time", hue="operator_intervention",  x_jitter=.1, y_jitter=.1, fit_reg=False)
    plt.show()

if __name__ == "__main__":
    main_2()
