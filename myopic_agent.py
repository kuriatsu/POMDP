#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import yaml
from ras_value_iteration_sweep import MDP
import sys

def myopic_policy(mdp, int_time=4, initial_state=[0, 14, 0, -1, 0.5, 0.5], intervention=0):

    index = tuple(mdp.to_index(initial_state))
    index_list = []
    policy_list = []
    buf = sorted(mdp.risk_positions)
    target_list = [mdp.risk_positions.tolist().index(v) for v in buf]
        
    while not mdp.final_state(index):
        decel_dist = (mdp.index_value(index, 1)**2 - 1.4**2)/(2*9.8*mdp.ordinary_G) + mdp.safety_margin
        int_request_dist = decel_dist + mdp.index_value(index, 1)*int_time
        
        if target_list and mdp.index_value(index, 2) >= int_time and mdp.index_value(index, 3) == target_list[0]:
            target_list.pop(0)    

        if target_list:
            dist_to_target = mdp.risk_positions[target_list[0]] - mdp.index_value(index, 0)
            if dist_to_target > int_request_dist:
                policy = -1
            elif target_list:
                policy = target_list[0]
        else:
            policy = -1
        print("closest target", policy, mdp.index_value(index, 0), int_request_dist)
        # if no target or further than intervention request timing
        # if closest_target is None or int_request_dist > min_dist:
        #    p = -1
        # in the intervention request distance 
        # elif int_request_dist <= min_dist:
        #     p = closest_target
        # else:
        #     p = mdp.index_value(index, 3)
            
        # policy = p[i] 
        [prob, index_after] =  mdp.state_transition(policy, index)[intervention]
        # print(mdp.action_value(policy, index)) # need to execute init_state_space
        index_list.append(index_after)
        policy_list.append(int(policy))
        index = tuple(index_after)

    return index_list, policy_list

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    index_list, policy_list = myopic_policy(MDP(param))
    print(index_list)
    print(policy_list)
