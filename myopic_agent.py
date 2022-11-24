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
    i = 0
    while not mdp.final_state(index):
        closest_target = None
        min_dist = mdp.prediction_horizon
        for i, p in enumerate(mdp.risk_positions):
            if mdp.index_value(index, 2) >= int_time and mdp.index_value(index, 3) == i:
                continue
            d = p - mdp.index_value(index, 0)
            if d < min_dist:
                min_dist = d
                closest_target = i 
        
        decel_dist = (mdp.index_value(index, 1)**2 - 1.4**2)/(2*9.8*mdp.ordinary_G) + mdp.safety_margin
        int_request_dist = decel_dist + mdp.index_value(index, 1)*int_time
        # if no target or further than intervention request timing
        if closest_target is None or int_request_dist > min_dist:
            p = -1
        # in the intervention request distance 
        elif int_request_dist <= min_dist:
            p = closest_target
        else:
            p = mdp.index_value(index, 3)
            
        # policy = p[i] 
        policy = p
        [prob, index_after] =  mdp.state_transition(policy, index)[intervention]
        # print(mdp.action_value(policy, index)) # need to execute init_state_space
        index_list.append(index_after)
        policy_list.append(int(policy))
        index = tuple(index_after)
        i += 1

    return index_list, policy_list

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    index_list, policy_list = myopic_policy(MDP(param))
    print(index_list)
    print(policy_list)
