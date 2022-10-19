#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import itertools

class MDP:
    def __init__(self):
        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0.0, 0.0]).T
        self.ego_state_max = np.array([200.0, 60.0]).T
        self.ego_state_width = np.array([10.0, 10.0]).T
 
        self.risk_pose = np.array([10.0, 80.0, 180.0]).T
        self.risk_difficulty = np.array([1.0, 1.0, 1.0]).T
        self.risk_confidence = np.array([80, 50, 10]).T
        # risk states: reach time
        self.risk_state_min = np.array([0.0]*len(self.risk_pose)).T
        self.risk_state_max = np.array([5.0]*len(self.risk_pose)).T
        self.risk_state_width = np.array([1.0]*len(self.risk_pose)).T
        
        # operator state: slope of time-acc of easy, middle, hard task 
        self.operator_state_min = np.array([0.0, 0.0, 0.0]).T
        self.operator_state_max = np.array([10.0, 10.0, 10.0]).T
        self.operator_state_width = np.array([1.0, 1.0, 1.0]).T

        self.state_min = np.r_[self.ego_state_min, self.risk_state_min, self.operator_state_min]
        self.state_max = np.r_[self.ego_state_max, self.risk_state_max, self.operator_state_max]
        self.state_width = np.r_[self.ego_state_width, self.risk_state_width, self.operator_state_width]

        self.index_nums = ((self.state_min - self.state_max)/self.state_width).astype(int)
        self.index_matrix = tuple(range(x) for x in self.index_nums)
        self.indexes = list(itertools.product(*self.index_matrix))
        # 0: not request intervention, 1:request intervention
        self.actions = np.array([0, 1])

        self.value_function, self.final_state_flag = self.init_value_function()
        self.policy = self.init_policy()
        self.goal_value = 100
        self.state_transition_probs = self.init_state_transition_probs(time_interval)


    def init_value_function(self):
        f = np.empty(self.index_nums)
        v = np.zeros(self.index_nums)
        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v, f

    def final_state(self, index):
        return index[0] == self.index_nums[0]


    def init_policy(self):
        p = np.zeros(self.index_nums)
        for index in self.indexes:
            p[index] = 0

        return p
