#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import itertools
import pickle
import copy

class MDP:
    def __init__(self):
        self.delta_t = 1.0

        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0, 0]).T
        self.ego_state_max = np.array([200, 60]).T
        self.ego_state_width = np.array([10, 10]).T

        # operator state: intercept_time, intercept_acc, slope
        self.operator_performance_min = np.array([0, 0.0, 0.0]).T 
        self.operator_performance_max = np.array([6, 1.0, 1.0).T 
        self.operator_performance_width = np.array([1, 0.1, 0.1]).T

        # risks state: likelihood 0:norisk, 100:risk , eta 
        self.risk_state_min = np.tile(np.array([0.0, -5]).T, (len(self.risk_positions), 1))
        self.risk_state_max = np.tile(np.array([1.0, 5]).T, (len(self.risk_positions), 1))
        self.risk_state_width = np.tile(np.array([0.25, 1]).T, (len(self.risk_positions), 1))
        self.risk_state_len = len(self.risk_state_width) / len(self.risk_positions) 
        
        # intervention state: int_time, target
        self.int_state_min = np.array([0, -1]).T
        self.int_state_max = np.array([6, len(self.risk_state_width)]).T
        self.int_state_width = np.array([1, 1]).T

        self.state_min = np.r_[self.ego_state_min, self.operator_performance_min, self.int_state_min, self.risk_state_min]
        self.state_max = np.r_[self.ego_state_max, self.operator_performance_max, self.int_state_max, self.risk_state_max]
        self.state_width = np.r_[self.ego_state_width, self.operator_performance_width, self.int_state_width, self.risk_state_width]

        # map 
        self.risk_positions = np.array([10, 80, 180]).T
        self.risk_speed = np.array([1, 1, 0]).T 

        # indexes
        self.index_nums = ((self.state_min - self.state_max)/self.state_width).astype(int)
        self.index_matrix = tuple(range(x) for x in self.index_nums)
        self.indexes = list(itertools.product(*self.index_matrix))
        self.ego_state_index = 0
        self.operator_performance_index = len(self.ego_state_width)
        self.int_state_index = len(self.ego_state_width) + len(self.operator_performance_width)
        self.risk_state_index = len(self.ego_state_width) + len(self.operator_performance_width) + len(self.int_state_width)

        # -1: not request intervention, else:request intervention
        self.actions = np.append(np.arange(len(self.risk_positions)), -1).T

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
        for index_after, prob in self.state_transition(action, index):
            collision = False
            risk = 0.0
            # state reward
            for i in range(len(self.risk_positions)):
                target_index = self.risk_state_index + self.risk_state_len * i 
                # when passed the target with crossing risk with 10km/h or higher
                if index_value(index, self.ego_state_index) == self.risk_positions[i] and index_value(index, self.ego_state_index+1) >= 10:
                    risk += index_value(index, self.risk_state_index)

                # pet
                relative_dist = self.risk_positions[i] - index_value(index, self.ego_state_index)
                pet = abs(relative_dist / index_value(index, self.ego_state_index+1) - index_value(index, target_index+2))
                if relative_dist > 0 and pet < 1.5:
                    collision = True

            # speed chenge 10km/h per delta_t
            confort = (abs(index_value(index_after, self.ego_state_index+1) - index_value(index, self.ego_state_index+1)) > 10.0)
            # action reward 
            bad_int_request = False
            int_acc_reward = False
            # when change the intervention target, judge the action decision
            if index[self.int_state_index+1] != -1 and index_after[self.int_state_index+1] != index[self.int_state_index+1]:
                bad_int_request = (get_int_performance(index) is None)
                low_acc_reward = (get_int_performance(index) < 0.5)
            
            action_value = -10000*collision -100*risk -10*confort -100*bad_int_request -10*low_acc_reward -self.delta_t + 1000*final_state(index_after)
            value += prob*action_value
            
        return value

    def state_transition(self, action, index):

        state_after = [index_value(index, i) for i in range(index)] 
        out_index_list = []

        # intervention state transition 
        if action[0] == index[self.int_state_index]:
            state_after[self.int_state_index+1] += self.delta_t
        else:
            state_after[self.int_state_index] = 0
            state_after[self.int_state_index+1] = action[0]

        # risk eta
        for i in range(len(self.positions)):
            eta_index = self.risk_state_index + i * self.risk_state_len + 1
            eta = index_value(eta_index)
            if self.risk_speed[i] > 0.0:
                eta_after =  eta - self.delta_t
            else:
                eta_after = eta

            eta_after = max(min(eta_after, 5.0), -5)
            state_after[eta_index] = eta_after

        # risk_state and ego_vehicle state 
        if index[self.int_state_index+1] != action[0] and index[self.int_state_index+1] != -1: 
            int_acc = get_int_performance(index)
            if int_acc is not None:
                target_index = self.risk_state_index + action[0]

                # transition if target is judged as risk
                int_prob = 0.5 
                buf_state_after = copy.deepcopy(state_after)
                buf_state_after[target_index] = int_acc 
                _, v, x = ego_vehicle_transition(state_after_risk)
                buf_state_after[self.ego_state_index] = x
                buf_state_after[self.ego_state_index+1] = v 
                out_index_list.append([int_prob, to_index(buf_state_after)]) 
                
                # transition if target is judged as norisk
                transition_prob = 0.5 
                buf_state_after = copy.deepcopy(state_after)
                buf_state_after[target_index] = 1.0 - int_acc 
                _, v, x = ego_vehicle_transition(state_after_risk)
                buf_state_after[self.ego_state_index] = x
                buf_state_after[self.ego_state_index+1] = v 
                out_index_list.append([int_prob, to_index(buf_state_after)]) 

        else:
            transition_prob = 1.0
            _, v, x = ego_vehicle_transition(state_after)
            state_after[self.ego_state_index] = x
            state_after[self.ego_state_index+1] = v 
            out_index_list.append([transition_prob, to_index(state_after)]) 
            
        return out_index_list


    def ego_vehicle_transition(self, index):
        # closest object
        closest_target_dist = self.ego_state_max[0]
        closest_target = None
        current_v = index_value(index, self.ego_state_index+1)
        current_pose = index_value(index, self.ego_state_index) 
        
        # get target for deceleration
        for i, pose in enumerate(self.risk_positions):
            dist = pose - current_pose
            if dist > 0.0 and dist < closest_target_dist and index_value(index, risk_state_index) >= 0.5:
                closest_target_dist = dist
                closest_target = i

        if closest_target is None:
            a = 0.2*9.8 if current_v < self.ideal_speed else -0.2*9.8
        else:
            deceleration_distance = (10**2-current_v**2)/(2*9.8*0.3)
            if closest_target_dist > deceleration_distance:
                a = 0.2*9.8 if current_v < self.ideal_speed else -0.2*9.8
            else:
                a = (10**2-current_v**2)/(2*9.8*closest_target_dist)

        v = ego_v + a * self.delta_t
        x = current_pose + ego_v * self.delta_t + 0.5 * a * self.delta_t**2

        return a, v, x
               

    def to_index(self, state):
        out_index = []
        for value, width, v_min, v_max in map(state, self.state_width, self.state_min, self.state_max):
            out_index.append(min(max(value, v_min), v_max)//width)

        return out_index


    def index_value(self, index, i):
        return index[i] * self.state_width[i]


    def get_int_performance(self, index):
        int_time = index_value(index, self.int_state_index) 
        intercept_time = index_value(index, self.operator_performance_index)
        if int_time < intercept_time:
            acc = None
        else:
            slope = index_value(self.operator_performance_index_value+2)
            intercept_acc = index_value(self.operator_performance_index_value+1)
            intercept_time = index_value(self.operator_performance_index_value)
            acc = intercept_acc + slope*(int_time-intercept_time)
            acc = min(acc, 1.0)
        
        return acc
        

def trial_until_sat():
    dp = MDP()
    delta = 1e100
    counter = 0
    while delta > 0.01:
        delta = dp.value_iteration_sweep()
        counter += 1
        print(counter, delta)

    with open("policy.txt", "w") as f:
        pickle.dump(dp.policy, f)

    with open("value.txt", "w") as f:
        pickle.dump(dp.value_function, f)


