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

        # operator state: intercept_time, intercept_acc, slope
        self.operator_performance_min = np.array([0.0, 0.0, 0.0]).T 
        self.operator_performance_max = np.array([6.0, 1.0, 10.0]).T 
        self.operator_performance_width = np.array([1.0, 0.1, 1.0]).T

        # intervention time
        self.int_time_min = np.array([0.0]).T
        self.int_time_max = np.array([6.0]).T
        self.int_time_width = np.array([1.0]).T

        # risks state: likelihood, eta 
        self.risk_state_min = np.tile(np.array([0, -5.0]).T, (len(self.risk_positions), 1))
        self.risk_state_max = np.tile(np.array([100, 5.0]).T, (len(self.risk_positions), 1))
        self.risk_state_width = np.tile(np.array([10, 1.0]).T, (len(self.risk_positions), 1))
        self.risk_state_len = len(self.risk_state_width) / len(self.risk_positions) 
        
        self.state_min = np.r_[self.ego_state_min, self.operator_performance_min, self.int_time_min, self.risk_state_min]
        self.state_max = np.r_[self.ego_state_max, self.operator_performance_max, self.int_time_max, self.risk_state_max]
        self.state_width = np.r_[self.ego_state_width, self.operator_performance_width, self.int_time_width, self.risk_state_width]

        # map 
        self.risk_positions = np.array([10.0, 80.0, 180.0]).T
        self.risk_speed = np.array([1.0, 1.0, 0.0]).T 

        # indexes
        self.index_nums = ((self.state_min - self.state_max)/self.state_width).astype(int)
        self.index_matrix = tuple(range(x) for x in self.index_nums)
        self.indexes = list(itertools.product(*self.index_matrix))
        self.ego_state_index = 0
        self.operator_performance_index = len(self.ego_state_width)
        self.int_time_index = len(self.ego_state_width) + len(self.operator_performance_width)
        self.risk_state_index = len(self.ego_state_width) + len(self.operator_performance_width) + len(self.int_time_width)

        # 0: not request intervention, 1:request intervention
        self.actions = np.array([0, 1])

        self.value_function, self.final_state_flag = self.init_value_function()
        self.policy = self.init_policy()
        self.goal_value = 100
        self.state_transition_probs = self.init_state_transition_probs(time_interval)

        # other params
        self.safety_margin = 10.0
        self.ideal_speed = 50.0


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

    
    def value_iteration_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                max_q = -1e100
                max_a = None
                qs = [self.action_value(a, index) for a in self.action]
                max_q = max(qs)
                max_a = self.actions[np.argmax(qs)]

                delta = abs(self.value_function[index] - max_q)
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = max_q
                self.policy[index] = np.array(max_a).T

        return max_delta

    def action_value(self, action, index):
        value = 0.0
        for delta, prob in self.state_transition(action, index):
            after = index + delta
            target_index = self.risk_state_index + self.risk_state_len * action[1]
            collision_reward = -100000 if index[self.ego_state_index] * abs(self.state_width[self.ego_state_index] - self.risk_positions[action[1]]) < 10 and index[target_index+2] < 1.0 else 0
            confort_reward = -100 if abs(delta[self.ego_state_index+1] * self.state_width[self.ego_state_index+1]) > 10.0 else 0.0
            int_request_reward = -100 if index[self.int_time_index] < 

    def state_transition(self, action, index):

        state_delta = np.zeros((len(index), len(self.actions))) 
        state_prob = np.zeros((len(self.actions)))
        
        # int acc
        int_prob, int_acc = get_int_performance(index, action)
        
        # vehicle transition
        ego_pose = index[0] * self.ego_state_width[0]
        ego_v = index[1] * self.ego_state_width[1]
        risk_pose = self.risk_positions[action[1]] # action[request, target] 
        ego_pose_index = self.ego_state_index
        ego_v_index = self.ego_state_index + 1

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
        risk_likelihood_index = self.risk_state_len * action[1]
        risk_likelihood = index[self.risk_state_index+risk_likelihood_index] * self.risk_state_width[risk_likelihood_index]
        if action[0] == 0 or (action[0] == 1 and index[-1] == 0):
            risk_likelihood_after = risk_likelihood
        elif action[0] == 1 and index[-1] == 1:
            risk_likelihood_after = 50
        elif action[0] == 1 and index[-1] == 2:
            risk_likelihood_after = 90 

        state_delta[0, self.risk_state_index + risk_likelihood_index] = (risk_likelihood - risk_likelihood_after) / self.state_width[risk_likelihood_index]
        state_delta[1, self.risk_state_index + risk_likelihood_index] = (risk_likelihood - risk_likelihood_after) / self.state_width[risk_likelihood_index]
        
        # risk eta
        for i in range(len(self.positions)):
            eta_index = self.risk_state_index + i * self.risk_state_len + 1
            eta = index[eta_index] * self.state_width[eta_index]
            if self.speed[i] > 0.0:
                eta_after =  eta - self.delta_t
            else:
                eta_after = eta

            eta_after = max(min(eta_after, 5.0), -5)
            state_delta[0, eta_index + i] = (eta_after - eta)/self.state_width[eta_index]
            state_delta[1, eta_index + i] = (eta_after - eta)/self.state_width[eta_index]

        if action[0] == 0:
            state_prob[1] = 0.0
        else:
            state_prob[1] = self.int_prob[index[-1]] 
       
        state_prob[0] = 1.0 - state_prob[1]
        out_data = []
        for delta, prob in map(state_delta, state_prob):
            out_data.append((delta, prob))

        return out_data


    def get_int_performance(self, index, action):
        int_time = index[self.int_time_index] * self.int_time_width[0] 
        intercept_time = index[self.operator_performance_index] * self.operator_performance_width[0]
        # dist_to_target = self.risk_positions[action[1]] - self.index[self.ego_index_index] * self.ego_index_width[0]
        # decel_dist = (index[self.ego_index_index+1]*self.ego_index_width[1])**2/(2*9.8*0.2)
        # time_to_decel = (dist_to_target - decel_dist) / (self.index[self.ego_index_index+1] * self.ego_index_width[1])
        # avairable_int_time = int_time + time_to_decel if time_to_decel > 0.0 else int_time 
        if int_time < intercept_time:
            prob = 0.0
            acc = 0.0
        else:
            prob = 1.0
            slope = index[self.operator_performance_index+2]*self.operator_performance_width[2]
            intercept_acc = index[self.operator_performance_index+1]*self.operator_performance_width[1]
            intercept_time = index[self.operator_performance_index]*self.operator_performance_width[0]
            acc = intercept_acc + slope*(int_time-intercept_time)
            acc = min(acc, 1.0)
        
        return prob, acc
        
