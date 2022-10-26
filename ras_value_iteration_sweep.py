#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import itertools
import pickle
import copy
import seaborn as sns
import matplotlib.pyplot as plt

class MDP:
    def __init__(self):
        self.delta_t = 1.0
        # other params
        self.goal_value = 100
        self.safety_margin = 10.0
        self.ideal_speed = 50.0
        # map 
        self.risk_positions = np.array([100,120]).T
        self.risk_speed = np.array([1, 0]).T 

        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0, 0]).T
        self.ego_state_max = np.array([200, 50]).T
        self.ego_state_width = np.array([20, 10]).T

        # operator state: intercept_time, intercept_acc, slope
        self.operator_performance_min = np.array([0, 0.0, 0.0]).T 
        self.operator_performance_max = np.array([6, 1.0, 1.0]).T 
        self.operator_performance_width = np.array([2, 0.25, 0.25]).T

        # risks state: likelihood 0:norisk, 100:risk , eta 
        self.risk_state_min = np.array([0.0, 0]*len(self.risk_positions)).T
        self.risk_state_max = np.array([1.0, 5]*len(self.risk_positions)).T
        self.risk_state_width = np.array([0.25, 1]*len(self.risk_positions)).T
        self.risk_state_len = int(len(self.risk_state_width) / len(self.risk_positions))
        
        # intervention state: int_time, target
        self.int_state_min = np.array([0, -1]).T
        self.int_state_max = np.array([6, len(self.risk_positions)]).T
        self.int_state_width = np.array([2, 1]).T

        self.state_min = np.r_[self.ego_state_min, self.operator_performance_min, self.int_state_min, self.risk_state_min]
        self.state_max = np.r_[self.ego_state_max, self.operator_performance_max, self.int_state_max, self.risk_state_max]
        self.state_width = np.r_[self.ego_state_width, self.operator_performance_width, self.int_state_width, self.risk_state_width]


        # indexes
        self.index_nums = ((self.state_max - self.state_min)/self.state_width).astype(int)
        print(self.index_nums)
        self.indexes = list(itertools.product(*tuple(range(x) for x in self.index_nums)))
        self.ego_state_index = 0
        self.operator_performance_index = len(self.ego_state_width)
        self.int_state_index = len(self.ego_state_width) + len(self.operator_performance_width)
        self.risk_state_index = len(self.ego_state_width) + len(self.operator_performance_width) + len(self.int_state_width)

        # -1: not request intervention, else:request intervention
        self.actions = np.append(np.arange(len(self.risk_positions)), -1).T

        self.value_function, self.final_state_flag = self.init_value_function()
        print("finish init")
        self.policy = self.init_policy()


    def init_value_function(self):
        f = np.empty(self.index_nums)
        v = np.zeros(self.index_nums)
        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal_value if f[index] else -100.0

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
            if not self.final_state_flag[index]:
                max_q = -1e100
                max_a = None
                qs = [self.action_value(a, index) for a in self.actions]
                max_q = max(qs)
                max_a = self.actions[np.argmax(qs)]
                delta = abs(self.value_function[index] - max_q)
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = max_q
                self.policy[index] = np.array(max_a).T

        return max_delta

    def action_value(self, action, index):
        value = 0.0
        for prob, index_after in self.state_transition(action, index):
            collision = False
            risk = 0.0
            # state reward
            for i in range(len(self.risk_positions)):
                target_index = self.risk_state_index + self.risk_state_len * i 
                # when passed the target with crossing risk with 10km/h or higher
                if self.index_value(index, self.ego_state_index) == self.risk_positions[i] and self.index_value(index, self.ego_state_index+1) >= 10:
                    risk += self.index_value(index, self.risk_state_index)

                # pet
                relative_dist = self.risk_positions[i] - self.index_value(index, self.ego_state_index)
                pet = abs(relative_dist / (self.index_value(index, self.ego_state_index+1)+1e-100) - self.index_value(index, target_index+1))
                if relative_dist > 0 and pet < 1.5:
                    collision = True

            # speed chenge 10km/h per delta_t
            confort = (abs(self.index_value(index_after, self.ego_state_index+1) - self.index_value(index, self.ego_state_index+1)) > 10.0)
            # action reward 
            bad_int_request = False
            int_acc_reward = False
            # when change the intervention target, judge the action decision
            if index[self.int_state_index+1] != -1 and index_after[self.int_state_index+1] != index[self.int_state_index+1]:
                int_acc = self.get_int_performance(index)
                bad_int_request = int_acc is None
                int_acc_reward = int_acc is not None and self.get_int_performance(index) < 0.5
            
            action_value = -10000*collision -100*risk -10*confort -100*bad_int_request -10*int_acc_reward -10*self.delta_t + 1000*self.final_state(index_after)
            value += prob*action_value
            
        return value

    def state_transition(self, action, index):

        state_value = [self.index_value(index, i) for i in index] 
        out_index_list = []

        # intervention state transition 
        if action == index[self.int_state_index]:
            state_value[self.int_state_index+1] += self.delta_t
        else:
            state_value[self.int_state_index] = 0
            state_value[self.int_state_index+1] = action

        # risk eta
        for i in range(len(self.risk_positions)):
            eta_index = int(self.risk_state_index + i * self.risk_state_len + 1)
            eta = self.index_value(index, eta_index)
            if self.risk_speed[i] > 0.0:
                eta_after =  eta - self.delta_t
            else:
                eta_after = eta

            eta_after = max(min(eta_after, 5.0), -5)
            state_value[eta_index] = eta_after

        # risk_state and ego_vehicle state 
        if index[self.int_state_index+1] != action and index[self.int_state_index+1] != -1: 
            int_acc = self.get_int_performance(index)
            if int_acc is not None:
                target_index = self.risk_state_index + action

                # transition if target is judged as risk
                int_prob = 0.5 
                buf_state_value = copy.deepcopy(state_value)
                buf_state_value[target_index] = int_acc 
                _, v, x = self.ego_vehicle_transition(buf_state_value)
                buf_state_value[self.ego_state_index] = x
                buf_state_value[self.ego_state_index+1] = v 
                out_index_list.append([int_prob, self.to_index(buf_state_value)]) 
                
                # transition if target is judged as norisk
                transition_prob = 0.5 
                buf_state_value = copy.deepcopy(state_value)
                buf_state_value[target_index] = 1.0 - int_acc 
                _, v, x = self.ego_vehicle_transition(buf_state_value)
                buf_state_value[self.ego_state_index] = x
                buf_state_value[self.ego_state_index+1] = v 
                out_index_list.append([int_prob, self.to_index(buf_state_value)]) 

        else:
            transition_prob = 1.0
            _, v, x = self.ego_vehicle_transition(state_value)
            state_value[self.ego_state_index] = x
            state_value[self.ego_state_index+1] = v 
            out_index_list.append([transition_prob, self.to_index(state_value)]) 
            
        return out_index_list


    def ego_vehicle_transition(self, state):
        # closest object
        closest_target_dist = self.ego_state_max[0]
        closest_target = None
        current_v = state[self.ego_state_index+1]/3.6
        current_pose = state[self.ego_state_index] 
        
        # get target for deceleration
        for i, pose in enumerate(self.risk_positions):
            dist = pose - current_pose
            if dist > 0.0 and dist < closest_target_dist and state[self.risk_state_index] >= 0.5:
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

        v = (current_v + a * self.delta_t) * 3.6
        x = current_pose + current_v * self.delta_t + 0.5 * a * self.delta_t**2

        return a, v, x
               

    def to_index(self, state):
        out_index = []
        for i in range(len(state)):
            out_index.append(min(max(state[i], self.state_min[i]), self.state_max[i])//self.state_width[i])

        return out_index


    def index_value(self, index, i):
        return index[i] * self.state_width[i]


    def get_int_performance(self, index):
        int_time = self.index_value(index, self.int_state_index) 
        intercept_time = self.index_value(index, self.operator_performance_index)
        if int_time < intercept_time:
            acc = None
        else:
            slope = self.index_value(index, self.operator_performance_index+2)
            intercept_acc = self.index_value(index, self.operator_performance_index+1)
            intercept_time = self.index_value(index, self.operator_performance_index)
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

    with open("policy.pkl", "wb") as f:
        pickle.dump(dp.policy, f)

    with open("value.pkl", "wb") as f:
        pickle.dump(dp.value_function, f)

    v = dp.value_function[:, :, 1, 3, 1, 2, 1, 1, 2]
    sns.heatmap(np.rot90(v), square=False)
    plt.show()

if __name__ == "__main__":
    trial_until_sat()
