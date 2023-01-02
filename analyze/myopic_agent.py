#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import yaml
from ras_value_iteration_sweep import MDP
import sys

def myopic_policy(mdp, int_time, value, initial_state, intervention_list):

    index = tuple(mdp.to_index(initial_state))
    indexes = []
    policies = []
    buf = sorted(mdp.risk_positions)
    target_list = [mdp.risk_positions.tolist().index(v) for v in buf]
        
    # cumlative_risk = 0
    travel_time = 0
    reward = 0

    while not mdp.final_state(index):
        # decel_dist = (5.6**2 - 1.4**2)/(2*9.8*mdp.ordinary_G)
        decel_dist = (mdp.index_value(index, 1)**2 - 1.4**2)/(2*9.8*mdp.ordinary_G)
        int_request_dist = decel_dist + mdp.index_value(index, 1)*int_time + mdp.safety_margin + 8
        intervention = intervention_list[target_list[0]] if target_list else 0
        reward += value[index]
        
        if target_list and mdp.index_value(index, 2) >= int_time and mdp.index_value(index, 3) == target_list[0]:
            target_list.pop(0)    

        if target_list:
            dist_to_target = mdp.risk_positions[target_list[0]] - mdp.index_value(index, 0)
            if mdp.index_value(index, 3) != -1 and mdp.index_value(index, 2) < int_time:
                policy = mdp.index_value(index, 3) 
            # elif dist_to_target > int_request_dist:
            #     policy = -1
            elif target_list:
                policy = target_list[0]
            # print(mdp.index_value(index, 1), dist_to_target, decel_dist, int_request_dist, target_list[0], policy)

        else:
            policy = -1
   
        index_after_list =  mdp.state_transition(policy, index)
        int_acc_prob_list = mdp.operator_model.get_acc_prob(mdp.index_value(index, mdp.int_state_index))
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
        

        # for i, risk_position in enumerate(mdp.risk_positions):
        #     pos = mdp.index_value(index, 0)
        #     pos_after = mdp.index_value(index_after, 0)
        #     speed = mdp.index_value(index, 1)
        #     if pos <= risk_position < pos_after and speed > mdp.min_speed:
        #         cumlative_risk += (0.5 - abs(mdp.index_value(index_after, mdp.risk_state_index+i) - 0.5))*2

        # print(mdp.action_value(policy, index)) # need to execute init_state_space
        indexes.append(index_after)
        policies.append(int(policy))
        index = tuple(index_after)

    travel_time = len(indexes)*1
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

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    index_list, policy_list, efficiency, risk = myopic_policy(MDP(param), 5, [0, 14, 0, -1, 0.75, 0.5], [-1, 0])
    print(index_list)
    print(policy_list)
