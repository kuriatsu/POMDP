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
# from w_perf.ras_value_iteration_sweep import MDP
# from wo_perf.ras_value_iteration_sweep_wo_perf import MDP
# from viased.ras_value_iteration_sweep_vias_intprob_huge import MDP
from myopic_agent import myopic_policy
import sys
import random

risk_colors = ["red", "blue", "green"]
trajectory_color = "green"

# mdp.init_state_space()
def plot(mdp, ax, indexes, policies, reward, risk_num, cumlative_risk, travel_time, request_time, title):
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
    for index in indexes:
        target = int(mdp.index_value(index, mdp.int_state_index+1))
        if target != -1:
            ax_risk.plot(mdp.index_value(index, 0), 0.5, c=risk_colors[target], marker="x", label="int request")
        else:
            ax_risk.plot(mdp.index_value(index, 0), 0.5, c="grey", marker="x")


    ax.annotate("travel time: "+str(travel_time), xy=(10, 5), size=8)
    ax.annotate("request time: "+str(request_time), xy=(10, 4), size=8)
    ax.annotate("# of risk omissions: "+str(cumlative_risk), xy=(10, 3), size=8)
    ax.annotate("reward: "+str(reward), xy=(10, 2), size=8)
    ax.set_xlabel("distance [m]", fontsize=14) 
    ax.set_ylabel("speed [m/s]", fontsize=14) 
    ax.set_title(title, fontsize=14, y=-0.25)
    ax_risk.set_ylabel("risk probability", fontsize=14) 

def get_cumlative_risk(mdp, indexes):

    cumlative_risk_low = 0
    cumlative_risk_high = 0
    pos = mdp.index_value(indexes[0], 0)
    final_risks = [mdp.index_value(indexes[-1], i) for i in range(mdp.risk_state_index, mdp.risk_state_index+len(mdp.risk_positions))]
    for index in indexes:
        pos_prev = pos
        pos = mdp.index_value(index, 0)
        speed = mdp.index_value(index, 1)
        if speed == mdp.min_speed:
            continue
        for i, risk_pos in enumerate(mdp.risk_positions):
            if pos_prev <= risk_pos < pos and 0.25 <= abs(final_risks[i]-0.5) < 0.5:
                cumlative_risk_low +=1
            if pos_prev <= risk_pos < pos and 0.0 <= abs(final_risks[i]-0.5) <  0.25:
                cumlative_risk_high +=1

    return cumlative_risk_low, cumlative_risk_high


def pomdp_agent(mdp, policy, value, initial_state, intervention_list):
    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policies = []
    reward = 0

    while not mdp.final_state(index):
        p = policy[index]
        reward += value[index]
        # get intervention or not from list if request state is not -1
        intervention = 0 if mdp.index_value(index, mdp.int_state_index+1) == -1 else intervention_list[int(mdp.index_value(index, mdp.int_state_index+1))]
        index_after_list =  mdp.state_transition(p, index)
        int_acc_prob_list = mdp.operator_model.get_acc_prob(mdp.index_value(index, mdp.int_state_index))

        # print("int", intervention, "candidate", index_after_list) 
        # if deterministic operator performance, index_after_list has 1 list when no intervention
        if len(index_after_list) == 1:
            index_after = index_after_list[0][1]

        # if deterministic operator performance, index_after_list has 2 list when intervention
        elif len(index_after_list) == 2:
            # select judge as norisk if intervention==-1
            index_after = index_after_list[intervention][1] 

        # no intervention request, all index is the same, prob is different. it should be changed 
        elif len(index_after_list) == 4:
            max_p = max([i[0] for i in index_after_list])
            index_after_index = [i for i, x in enumerate(index_after_list) if x[0]==max_p][0]
            index_after = index_after_list[index_after_index][1]

        # if intervention, index_after_index is 7 (stocastic performance) or 2 (deterministic performance)
        # select highest value policy from them 
        else:
            max_p = max([i[1] for i in int_acc_prob_list])
            index_after_index = [i[1] for i in int_acc_prob_list].index(max_p)
            # select None
            if index_after_index == len(int_acc_prob_list)-1:
                index_after = index_after_list[-1][1]
            # select judge as norisk
            elif intervention == -1:
                index_after = index_after_list[index_after_index*2+1][1] 
            # select judge as risk
            else:
                index_after = index_after_list[index_after_index*2][1] 

        indexes.append(index_after)
        policies.append(int(p))
        index = tuple(index_after)

    travel_time = len(indexes)
    request_time = 0
    request_count = 0
    last_p = -2 
    for p in policies:
        if p != -1:
            request_time += 1
        if p not in [-1, last_p]:
            request_count += 1
        last_p = p

    return indexes, policies, reward, travel_time, request_time, request_count


def egotistical_agent(mdp, value, initial_state, intervention_list):
    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policies = []
    reward = 0
    while not mdp.final_state(index):
        policy = -1 
        reward += value[index]

        intervention = 0 if mdp.index_value(index, mdp.int_state_index+1) == -1 else intervention_list[int(mdp.index_value(index, mdp.int_state_index+1))]
        index_after_list =  mdp.state_transition(policy, index)
        max_p = max([i[0] for i in index_after_list])
        highest_index_list = [i for i, x in enumerate(index_after_list) if x[0]==max_p]
        index_after = index_after_list[highest_index_list[intervention]][1]
        
        indexes.append(index_after)
        policies.append(int(policy))
        index = tuple(index_after)

    travel_time = len(indexes)
    request_time = len([p for p in policies if p!=-1])
    return indexes, policies, reward, travel_time, request_time, 0

def visualize_speed(scenario_list, dir):

    for scenario in scenario_list:
        fig = plt.figure(figsize=[6, 8])
        axes = fig.subplots(4, 1, sharex="all", sharey="all")
        # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.90, top=0.95, wspace=0.3, hspace=0.3)
        intervention_list = scenario[2]
        initial_state = scenario[1]
        param_file = scenario[0]
        with open(dir+param_file) as f:
            param = yaml.safe_load(f)


        mdp = MDP(param)
        filename = param_file.split("/")[-1].split(".")[0]
        print(dir+param_file)
        with open(dir+f"{filename}_p.pkl", "rb") as f:
            p = pickle.load(f)
        with open(dir+f"{filename}_v.pkl", "rb") as f:
            v = pickle.load(f)


        indexes, policies, reward, travel_time, request_time, request_count = pomdp_agent(mdp, p, v, initial_state, intervention_list)
        cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
        plot(mdp, axes[0], indexes, policies, reward, len(mdp.risk_positions), cumlative_risk_low+cumlative_risk_high, travel_time, request_time, "full-model POMDP")
        vel_p = 0
        a_list = []
        for vel in [mdp.index_value(index, 1) for index in indexes]:
            a_list.append(vel - vel_p)
            vel_p = vel
                
        indexes, policies, reward, travel_time, request_time, request_count = myopic_policy(mdp, 5, v, initial_state, intervention_list)
        cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
        plot(mdp, axes[1], indexes, policies, reward, len(mdp.risk_positions), cumlative_risk_low+cumlative_risk_high, travel_time, request_time, "myopic")

        indexes, policies, reward, travel_time, request_time, request_count = egotistical_agent(mdp, v, initial_state, intervention_list)
        cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
        plot(mdp, axes[2], indexes, policies, reward, len(mdp.risk_positions), cumlative_risk_low+cumlative_risk_high, travel_time, request_time, "egoistical")

        filename_wo_perf = "param_"+filename.split("_")[1]+"_p.pkl"
        with open(f"wo_perf/{filename_wo_perf}", "rb") as f:
            p = pickle.load(f)
        indexes, policies, reward, travel_time, request_time, request_count = pomdp_agent(mdp, p, v, initial_state, intervention_list)
        cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
        plot(mdp, axes[3], indexes, policies, reward, len(mdp.risk_positions), cumlative_risk_low+cumlative_risk_high, travel_time, request_time, "POMDP w/o perf")

        plt.title(filename)
        plt.savefig(dir+f"{filename}_{str(scenario)}_speed.svg")
        # plt.show()

        # axes, fig = plt.subplots()
        # viz_v = eval("v" + param["visualize_elem"])
        # print(viz_v[60, :])
        # sns.heatmap(np.rot90(viz_v), square=False)
        # plt.title(f"{filename}value")
        # plt.savefig(f"{filename}_value.svg")
        # plt.show()

def simulation(initial_states, dir, param_list, out_file):

    # -1:no intervention 0:intervention
    intervention_lists = [
            [-1, -1], 
            [-1, 0], 
            [0, -1], 
            [0, 0], 
            ]
    result_list = pd.DataFrame(columns=["param", "risk", "position", "intervention", "agent", "cumlative_risk", "cumlative_risk_low", "cumlative_risk_high", "travel_time", "request_time", "request_count"])
    for initial_state in initial_states:
        for intervention_list in intervention_lists:
            for idx, param_file in enumerate(param_list):

                buf_list = pd.DataFrame(columns=result_list.columns)
                print(param_file)
                with open(dir+param_file) as f:
                    param = yaml.safe_load(f)

                mdp = MDP(param)
                filename = param_file.split("/")[-1].split(".")[0]
                with open(dir+f"{filename}_p.pkl", "rb") as f:
                    p = pickle.load(f)
                with open(dir+f"{filename}_v.pkl", "rb") as f:
                    v = pickle.load(f)

                # pomdp
                indexes, policies, reward, travel_time, request_time, request_count = pomdp_agent(mdp, p, v, initial_state, intervention_list)
                cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
                buf = pd.DataFrame([[
                    filename, 
                    str(initial_state[-2:]), 
                    str(mdp.risk_positions), 
                    str(intervention_list), 
                    "pomdp", 
                    cumlative_risk_low+cumlative_risk_high,
                    cumlative_risk_low, 
                    cumlative_risk_high,
                    travel_time, 
                    request_time,
                    request_count
                    ]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)

                # myopic agent
                indexes, policies, reward, travel_time, request_time, request_count = myopic_policy(mdp, 5, v, initial_state, intervention_list)
                cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
                buf = pd.DataFrame([[
                    filename, 
                    str(initial_state[-2:]), 
                    str(mdp.risk_positions), 
                    str(intervention_list), 
                    "myopic", 
                    cumlative_risk_low+cumlative_risk_high,
                    cumlative_risk_low, 
                    cumlative_risk_high,
                    travel_time, 
                    request_time,
                    request_count
                    ]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)

                # egoistical
                indexes, policies, reward, travel_time, request_time, request_count = egotistical_agent(mdp, v, initial_state, intervention_list)
                cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
                buf = pd.DataFrame([[
                    filename, 
                    str(initial_state[-2:]), 
                    str(mdp.risk_positions), 
                    str(intervention_list), 
                    "egoistical", 
                    cumlative_risk_low+cumlative_risk_high,
                    cumlative_risk_low, 
                    cumlative_risk_high,
                    travel_time, 
                    request_time,
                    request_count
                    ]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)

                # w/o perf pomdp
                filename_wo_perf = "param_"+filename.split("_")[1]+"_p.pkl"
                with open(f"wo_perf/{filename_wo_perf}", "rb") as f:
                    p = pickle.load(f)
                indexes, policies,reward,  travel_time, request_time, request_count = pomdp_agent(mdp, p, v, initial_state, intervention_list)
                cumlative_risk_low, cumlative_risk_high = get_cumlative_risk(mdp, indexes)
                buf = pd.DataFrame([[
                    filename, 
                    str(initial_state[-2:]), 
                    str(mdp.risk_positions), 
                    str(intervention_list), 
                    "pomdp_wo_perf", 
                    cumlative_risk_low+cumlative_risk_high,
                    cumlative_risk_low, 
                    cumlative_risk_high,
                    travel_time, 
                    request_time,
                    request_count
                    ]], columns=result_list.columns)
                buf_list = pd.concat([buf_list, buf], ignore_index=True)

                # concat all agent
                result_list = pd.concat([result_list, buf_list], ignore_index=True)

    result_list.to_csv(out_file)


def analyze(in_file, out_comp_file, out_summary_file, imp_rate_file):
    sns.set(context='paper', style='whitegrid')
    df = pd.read_csv(in_file)
    target_scenario_df = df 
    
    fig, axes = plt.subplots(3, 1)
    sns.histplot(data=target_scenario_df[target_scenario_df.agent.isin(["egoistical", "myopic", "pomdp"])], x="travel_time", hue="agent", hue_order=["egoistical", "myopic", "pomdp"], kde=True, ax=axes[0])
    sns.histplot(data=target_scenario_df[target_scenario_df.agent.isin(["egoistical", "myopic", "pomdp"])], x="request_time", hue="agent", hue_order=["egoistical", "myopic", "pomdp"], kde=True, ax=axes[1])
    sns.histplot(data=target_scenario_df[target_scenario_df.agent.isin(["egoistical", "myopic", "pomdp"])], x="cumlative_risk", hue="agent", hue_order=["egoistical", "myopic", "pomdp"], kde=True, ax=axes[2])
    plt.show()

    # comparison myopic vs pomdp
    comp_myopic_pomdp_df = pd.DataFrame(columns=[
        "param", 
        "initial_risk", 
        "intervention", 
        "cumlative_risk", 
        "travel_time", 
        "request_time", 
        ])
    for param in target_scenario_df.param.drop_duplicates():
        for intervention in target_scenario_df.intervention.drop_duplicates():
            for risk in target_scenario_df.risk.drop_duplicates():
                target_df = target_scenario_df[(target_scenario_df.risk == risk) & (target_scenario_df.intervention == intervention) & (target_scenario_df.param == param)]
                # initial_risk = ",".join(scenario.split(",")[:2])
                # operator_intervention = ",".join(scenario.split(",")[2:])
                # if target_df.intervention.iloc[-1] in ["[-1, 0]", "[0, -1]"]:
                #     continue
                cumlative_risk = target_df[target_df.agent=="pomdp"].cumlative_risk.iloc[-1] - target_df[target_df.agent=="myopic"].cumlative_risk.iloc[-1]
                travel_time = target_df[target_df.agent=="pomdp"].travel_time.iloc[-1] - target_df[target_df.agent=="myopic"].travel_time.iloc[-1]
                request_time = target_df[target_df.agent=="pomdp"].request_time.iloc[-1] - target_df[target_df.agent=="myopic"].request_time.iloc[-1]
                buf = pd.DataFrame([[
                    param, 
                    risk, 
                    intervention, 
                    cumlative_risk, 
                    travel_time, 
                    request_time, 
                    ]], columns=comp_myopic_pomdp_df.columns)
                # print(target_df)
                comp_myopic_pomdp_df = pd.concat([comp_myopic_pomdp_df, buf], ignore_index=True)
    comp_myopic_pomdp_df.to_csv(out_comp_file)

    g = sns.jointplot(data=comp_myopic_pomdp_df, x="travel_time", y="request_time", hue="cumlative_risk", s=50.0, alpha=1.0, ylim=(-12.5, 12.5), palette=["blue", "green", "gray", "orange", "red"], marginal_ticks=True)
    # g = sns.jointplot(data=comp_myopic_pomdp_df, x="travel_time", y="request_time", hue="cumlative_risk", s=50.0, alpha=1.0, ylim=(-12.5, 12.5), palette=["gray", "orange", "red"], marginal_ticks=True)
    g.plot(sns.scatterplot, sns.histplot)
    plt.show()

    # summary
    summary_df = pd.DataFrame(columns=[
        "agent", 
        "risk-low", 
        "risk-high", 
        "trav-mean", 
        "trav-std", 
        "req-mean", 
        "req-std",
        "req-min",
        "req-max",
        "req-cnt",
        "req-min",
        "req-max",
        ])
    for agent in target_scenario_df.agent.drop_duplicates():
        target_df = target_scenario_df[(target_scenario_df.agent == agent)]

        # initial_risk = ",".join(str(target_df.scenario).split(",")[:2])
        # operator_intervention = ",".join(str(target_df.scenario).split(",")[2:])

        ## remove unnecesary intervention
        # if target_df.intervention.str in ["[-1, 0]", "[0, -1]"]:
        #     continue
        buf = pd.DataFrame([[
            agent, 
            target_df.cumlative_risk_low.sum()/(len(target_df)*2),
            target_df.cumlative_risk_high.sum()/(len(target_df)*2),
            target_df.travel_time.mean(),
            target_df.travel_time.std(),
            target_df.request_time.mean(),
            target_df.request_time.std(),
            target_df.request_time.min(),
            target_df.request_time.max(),
            target_df.request_count.mean(),
            target_df.request_count.min(),
            target_df.request_count.max(),
            ]], columns=summary_df.columns)
        summary_df = pd.concat([summary_df, buf], ignore_index=True)

    summary_df.to_csv(out_summary_file)


    # 2 policies comparison
    imp_rate_df = pd.DataFrame(columns=["target-agent", "improve", "trav_imp", "req_imp", "same", "degrade"])
    target_agent_list = [["egoistical", "myopic"], ["egoistical", "pomdp"], ["myopic", "pomdp"]]
    for target, agent in target_agent_list:
        improve = 0
        trav_imp = 0
        req_imp = 0
        same = 0
        total = 0
        for idx, row in target_scenario_df[target_scenario_df.agent==agent].iterrows():
            target_df = target_scenario_df[
                    (target_scenario_df.agent==target) &
                    (target_scenario_df.risk == row.risk) &
                    (target_scenario_df.position == row.position) &
                    (target_scenario_df.intervention == row.intervention)
                    ].iloc[-1]
            improve += ((row.travel_time < target_df.travel_time and row.request_time < target_df.request_time) or
                        (row.travel_time < target_df.travel_time and row.request_time < target_df.request_time)) 
            trav_imp += (row.travel_time < target_df.travel_time and row.request_time >= target_df.request_time)
            req_imp +=  (row.travel_time >= target_df.travel_time and row.request_time < target_df.request_time)
            same +=  (row.travel_time == target_df.travel_time and row.request_time == target_df.request_time)
            total += 1

        buf_df = pd.DataFrame([[f"{target}-{agent}", improve/total, trav_imp/total, req_imp/total, same/total, (total-improve-trav_imp-req_imp-same)/total]], columns=imp_rate_df.columns)
        imp_rate_df = pd.concat([imp_rate_df, buf_df], ignore_index=True)
    imp_rate_df.to_csv(imp_rate_file)        


if __name__ == "__main__":
    ####################################
    # simulation
    ####################################
    param_list = [
            "param_1_low.yaml",
            "param_2_low.yaml",
            "param_3_low.yaml",
            "param_4_low.yaml",
            "param_5_low.yaml",
            "param_6_low.yaml",
            "param_7_low.yaml",
            "param_8_low.yaml",
            "param_9_low.yaml",
            "param_10_low.yaml",
            "param_11_low.yaml",
            "param_12_low.yaml",
            "param_13_low.yaml",
            "param_14_low.yaml",
            "param_15_low.yaml",
            "param_16_low.yaml",
            "param_17_low.yaml",
            "param_18_low.yaml",
            "param_19_low.yaml",
            "param_20_low.yaml",
            "param_21_low.yaml",
            ]
    initial_states = [
            [0, 11.2, 0, -1, 0.0, 0.0],
            [0, 11.2, 0, -1, 0.25, 0.0],
            [0, 11.2, 0, -1, 0.0, 0.25],
            [0, 11.2, 0, -1, 0.25, 0.25],
            
            [0, 11.2, 0, -1, 0.5, 0.0],
            [0, 11.2, 0, -1, 0.5, 0.25],
            [0, 11.2, 0, -1, 0.75, 0.0],
            [0, 11.2, 0, -1, 0.75, 0.25],
            [0, 11.2, 0, -1, 1.0, 0.0],
            [0, 11.2, 0, -1, 1.0, 0.25],

            [0, 11.2, 0, -1, 0.0, 0.5],
            [0, 11.2, 0, -1, 0.0, 0.75],
            [0, 11.2, 0, -1, 0.0, 1.0],
            [0, 11.2, 0, -1, 0.25, 0.5],
            [0, 11.2, 0, -1, 0.25, 0.75],
            [0, 11.2, 0, -1, 0.25, 1.0],

            [0, 11.2, 0, -1, 0.5, 0.5],
            [0, 11.2, 0, -1, 0.5, 0.75],
            [0, 11.2, 0, -1, 0.5, 1.0],
            [0, 11.2, 0, -1, 0.75, 0.5],
            [0, 11.2, 0, -1, 0.75, 0.75],
            [0, 11.2, 0, -1, 0.75, 1.0],
            [0, 11.2, 0, -1, 1.0, 0.5],
            [0, 11.2, 0, -1, 1.0, 0.75],
            [0, 11.2, 0, -1, 1.0, 1.0],
            ]
    # simulation(initial_states, "./", param_list, "result_low.csv")

    ####################################
    # comparison
    ####################################
    # analyze("result_low.csv", "result_comparison_low.csv", "result_summary_low.csv", "result_summary_rate_low.csv")
    # analyze("result_mid.csv", "result_comparison_mid.csv", "result_summary_mid.csv", "result_summary_rate_mid.csv")
    # analyze("result_low.csv", "result_comparison_low.csv", "result_summary_low.csv", "result_summary_rate_low.csv")

    ####################################
    # visualization
    ####################################
    # -1:judged as risk 0:no risk
    scenario_list = [
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            # ["param_9_mid_delta_t.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_9_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_9_low.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_12_low.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_9_high.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [-1, 0]],
            ["param_12_high.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            # ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            # ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],
            # ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.75], [0, -1]],
            # ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [0, -1]],
            # ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [-1, 0]],

            # both ok without risk
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ["param_16_mid.yaml", [0, 11.2, 0, -1, 0.25, 0.25], [0, -1]],
            ["param_16_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.25], [0, -1]],
            ["param_16_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [0, -1]],

            # both bad 
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, -1]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, -1]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [0, -1]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [0, 0]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, -1]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [0, -1]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [0, -1]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [0, -1]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [0, 0]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [0, 0]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [0, 0]],
            ["param_17_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [0, 0]],

            # trave good
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.25], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [-1, 0]],
            ["param_12_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.25], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [-1, 0]],
            ["param_13_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.25], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [-1, 0]],
            ["param_14_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.25], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.5], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.5, 0.75], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.25], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.5], [-1, 0]],
            ["param_18_mid.yaml", [0, 11.2, 0, -1, 0.75, 0.75], [-1, 0]],
            ]

    visualize_speed(scenario_list, "./")
