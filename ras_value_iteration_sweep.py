#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import itertools

class MDP:
    def __init__(self):
        self.delta_t = 1.0

        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0.0, 0.0]).T
        self.ego_state_max = np.array([200.0, 60.0]).T
        self.ego_state_width = np.array([10.0, 10.0]).T

        # map 
        self.risk_positions = np.array([10.0, 80.0, 180.0]).T
        self.risk_difficulty = np.array([1, 1, 1]).T # 1:middle 0:easy 2:hard
        self.risk_distance = np.array([10.0, 30.0, 5.0)].T
        self.risk_speed = np.array([1.0, 1.0, 0.0]).T 

        # risks variables 
        self.risk_likelihood_min = np.array([0]*len(self.risk_positions)).T
        self.risk_likelihood_max = np.array([100]*len(self.risk_positions)).T
        self.risk_likelihood_width = np.array([10]*len(self.risk_positions)).T
        self.risk_eta_min = np.array([0.0]*len(self.risk_positions)).T
        self.risk_eta_max = np.array([5.0]*len(self.risk_positions)).T
        self.risk_state_width = np.array([1.0]*len(self.risk_positions)).T
        
        # operator state: slope of time-acc of easy, middle, hard task 
        self.operator_performance_min = np.array([0]).T # 0:impossible to intervene 1:low acc 2:high acc
        self.operator_performance_max = np.array([2]).T 
        self.operator_performance_width = np.array([1]).T

        self.state_min = np.r_[self.ego_state_min, self.risk_likelihood_min, self.risk_eta_min, self.operator_performance_min]
        self.state_max = np.r_[self.ego_state_max, self.risk_likelihood_max, self.risk_eta_max, self.operator_performance_max]
        self.state_width = np.r_[self.ego_state_width, self.risk_likelihood_width, self.risk_eta_width, self.operator_performance_width]

        self.index_nums = ((self.state_min - self.state_max)/self.state_width).astype(int)
        self.index_matrix = tuple(range(x) for x in self.index_nums)
        self.indexes = list(itertools.product(*self.index_matrix))

        # 0: not request intervention, 1:request intervention
        self.actions = np.array([0, 1])

        self.value_function, self.final_state_flag = self.init_value_function()
        self.policy = self.init_policy()
        self.goal_value = 100
        self.state_transition_probs = self.init_state_transition_probs(time_interval)

        self.safety_margin = 10.0
        self.ideal_speed = 50.0
        self.int_prob = np.array([0.0, 0.5, 1.0])


    def init_value_function(self):
        f = np.empty(self.index_nums)
        v = np.zeros(self.index_nums)
        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v, f

    def final_state(self, index):
        return index[0] == float(self.index_nums[0]) - 1.0


    def init_policy(self):
        p = np.zeros(self.index_nums)
        for index in self.indexes:
            p[index] = 0

        return p

    def state_transition(self, action, index):

        state_delta = np.zeros((len(index), len(self.actions))) 
        state_prob = np.zeros((len(self.actions)))

        # vehicle transition
        ego_v = index[1] * self.ego_state_width[1] 
        ego_pose = index[0] * self.ego_state_width[0]
        risk_pose = self.risk_positions[action[1]] 

        # vehicle transition if no intervention
        a_noint = -v**2 / (2 * abs(risk_pose - ego_pose - self.safety_margin))
        ego_pose_noint = ego_v * self.delta_t + 0.5 * self.delta_t**2 *a_noint
        ego_v_noint = a_noint * self.delta_t
        state_delta[0, 0] = ego_pose_noint / self.state_width[0]
        state_delta[0, 1] = ego_v_noint / self.state_width[1]
        
        # vehicle transition if intervention
        a_int = 0.2*9.8 if ego_v < self.ideal_speed else -0.2*9.8
        ego_pose_int = ego_pose + ego_v * self.delta_t + 0.5 * self.delta_t**2 *a_int
        ego_v_int = ego_v + a_int * self.delta_t
        state_delta[1, 0] = ego_pose_int / self.state_width[0]
        state_delta[1, 1] = ego_v_int / self.state_width[1]
        
        # intervention changes risk likelihood
        risk_likelihood_index = 2+action[1]
        risk_likelihood = index[risk_likelihood_index] * self.risk_likelihood_width[action[1]]
        if action[0] == 0 or (action[0] == 1 and index[-1] == 0):
            risk_likelihood_after = risk_likelihood
        elif action[0] == 1 and index[-1] == 1:
            risk_likelihood_after = 50
        elif action[0] == 1 and index[-1] == 2:
            risk_likelihood_after = 90 

        state_delta[0, risk_likelihood_index] = (risk_likelihood - risk_likelihood_after) / self.state_width[risk_likelihood_index]
        state_delta[1, risk_likelihood_index] = (risk_likelihood - risk_likelihood_after) / self.state_width[risk_likelihood_index]
        
        # risk eta
        risk_eta_index = 2 + len(self.risk_positions)
        for i, eta in enumerate(index[eta_index:eta_index+len(self.risk_positions)]):
            if self.risk_speed[i] > 0.0:
                eta_after = eta - self.delta_t
            else:
                eta_after = eta
            state_delta[0, eta_index+i] = (eta_after - eta)/self.state_width[eta_index+i]
            state_delta[1, eta_index+i] = (eta_after - eta)/self.state_width[eta_index+i]

        if action[0] == 0:
            int_prob = 0.0
        else:
            int_prob = self.int_prob[index[-1]]

    def int_acc(self, state):

