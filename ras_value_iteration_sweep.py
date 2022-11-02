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
        self.delta_t = 2.0
        self.prediction_horizon = 100
        self.safety_margin = 10.0
        self.ideal_speed = 50.0
        self.min_speed = 10.0/3.6
        self.ordinary_G = 0.2
        self.min_step_size = self.prediction_horizon*3.6/self.ideal_speed
        self.discount_factor = self.min_step_size/(self.min_step_size+1.0)
        self.goal_value = 100

        # map 
        # self.risk_positions = np.array([80,160]).T
        # self.risk_speed = np.array([1, 0]).T 

        self.risk_positions = np.array([60]).T
        self.risk_speed = np.array([1]).T 

        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0, 0]).T
        self.ego_state_max = np.array([self.prediction_horizon, self.ideal_speed]).T
        self.ego_state_width = np.array([10, 10]).T

        # operator state: intercept_time, intercept_acc, slope
        self.operator_performance_min = np.array([0, 0.0, 0.0]).T 
        self.operator_performance_max = np.array([6, 1.0, 1.0]).T 
        self.operator_performance_width = np.array([2, 0.25, 0.25]).T

        # risks state: likelihood 0:norisk, 100:risk , eta 
        self.risk_state_min = np.array([0.0, -4]*len(self.risk_positions)).T
        self.risk_state_max = np.array([1.0, 4]*len(self.risk_positions)).T
        self.risk_state_width = np.array([0.25, 2]*len(self.risk_positions)).T
        self.risk_state_len = int(len(self.risk_state_width) / len(self.risk_positions))
        
        # intervention state: int_time, target
        self.int_state_min = np.array([0, -1]).T
        self.int_state_max = np.array([6, len(self.risk_positions)-1]).T
        self.int_state_width = np.array([2, 1]).T

        self.state_min = np.r_[self.ego_state_min, self.operator_performance_min, self.int_state_min, self.risk_state_min]
        self.state_max = np.r_[self.ego_state_max, self.operator_performance_max, self.int_state_max, self.risk_state_max]
        self.state_width = np.r_[self.ego_state_width, self.operator_performance_width, self.int_state_width, self.risk_state_width]

        self.index_nums=None
        self.indexes = None
        self.ego_state_index = None
        self.operator_performance_index = None
        self.int_state_index = None
        self.risk_state_index = None
        # 0: not request intervention, else:request intervention
        self.actions = np.arange(-1, len(self.risk_positions)).T
        self.value_function = None
        self.final_state_flag = None
        self.policy = None


    def init_state_space(self):

        # indexes
        self.index_nums = (1+(self.state_max - self.state_min)/self.state_width).astype(int)
        print(self.index_nums)
        self.indexes = list(itertools.product(*tuple(range(x) for x in self.index_nums)))
        print("indexes size", self.indexes.__sizeof__())
        self.ego_state_index = 0
        self.operator_performance_index = len(self.ego_state_width)
        self.int_state_index = len(self.ego_state_width) + len(self.operator_performance_width)
        self.risk_state_index = len(self.ego_state_width) + len(self.operator_performance_width) + len(self.int_state_width)

        self.value_function, self.final_state_flag = self.init_value_function()
        print("value_function size", self.value_function.__sizeof__())
        self.policy = self.init_policy()
        print("policy size", self.policy.__sizeof__())

    def init_value_function(self):
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)
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
                # print(qs)
                max_q = max(qs)
                max_a = self.actions[np.argmax(qs)]
                delta = abs(self.value_function[index] - max_q)
                max_delta = max(delta, max_delta)

                self.value_function[index] = max_q
                self.policy[index] = np.array(max_a).T

        return max_delta

    def action_value(self, action, index):
        value = 0.0
        # print(index[0], [self.index_value(index, v) for v in range(len(index))])
        for prob, index_after in self.state_transition(action, index):
            # print(index[0], prob, [self.index_value(index_after, v) for v in range(len(index))])
            collision = False
            risk = 0.0
            # state reward
            for i in range(len(self.risk_positions)):
                target_index = self.risk_state_index + self.risk_state_len * i 
                # when passed the target with crossing risk with 10km/h or higher
                if self.index_value(index_after, self.ego_state_index) == self.risk_positions[i] and self.index_value(index_after, self.ego_state_index+1) > self.min_speed and self.index_value(index_after, target_index) > 0.8:
                    collision = True
                # pet
                relative_dist = self.risk_positions[i] - self.index_value(index_after, self.ego_state_index)
                pet = abs(relative_dist / (self.index_value(index_after, self.ego_state_index+1)+1e-100) - self.index_value(index_after, target_index+1))
                if relative_dist > 0 and pet < 1.5 and self.index_value(index_after, self.ego_state_index+1) > self.min_speed:
                    risk += 1

            # speed chenge 10km/h per delta_t
            confort = (abs(self.index_value(index_after, self.ego_state_index+1) - self.index_value(index, self.ego_state_index+1))/(3.6*9.8*self.delta_t) > self.ordinary_G)
            # action reward 
            bad_int_request = False
            int_acc_reward = False
            int_reward = False if action == -1 else True
            # when change the intervention target, judge the action decision
            if self.index_value(index, self.int_state_index+1) not in  [-1, action]:
                int_acc = self.get_int_performance(index)
                bad_int_request = int_acc is None
                int_acc_reward = int_acc is not None and self.get_int_performance(index) < 0.5
            
            action_value = -10000*collision -10*risk -1*confort -100*bad_int_request -10*int_acc_reward -1*int_reward -1*self.delta_t + self.goal_value*self.final_state(index_after)
            value += prob * (self.value_function[tuple(index_after)] + action_value) * self.discount_factor
            
        return value

    def state_transition(self, action, index):

        state_value = [self.index_value(index, i) for i in range(len(index))] 
        out_index_list = []

        # intervention state transition 
        if action == self.index_value(index, self.int_state_index+1):
            state_value[self.int_state_index] += self.delta_t
        else:
            state_value[self.int_state_index] = 0
            state_value[self.int_state_index+1] = action

        # print("risk eta")
        for i in range(len(self.risk_positions)):
            eta_index = int(self.risk_state_index + i * self.risk_state_len + 1)
            eta = self.index_value(index, eta_index)
            if self.risk_speed[i] > 0.0:
                eta_after =  eta - self.delta_t
            else:
                eta_after = eta

            eta_after = max(min(eta_after, self.risk_state_max[1]), self.risk_state_min[1])
            state_value[eta_index] = eta_after

        # print("risk_state and ego_vehicle state") 
        int_acc = self.get_int_performance(index)
        if self.index_value(index, self.int_state_index+1) != action and self.index_value(index, self.int_state_index+1) != -1 and int_acc is not None : 
            target_index = self.risk_state_index + action * self.risk_state_len

            # print("transition if target is judged as risk")
            int_prob = 0.5 
            buf_state_value = copy.deepcopy(state_value)
            buf_state_value[target_index] = (1.0 + int_acc) * 0.5 
            _, v, x = self.ego_vehicle_transition(buf_state_value)
            buf_state_value[self.ego_state_index] = x
            buf_state_value[self.ego_state_index+1] = v 
            out_index_list.append([int_prob, self.to_index(buf_state_value)]) 
            
            # print("transition if target is judged as norisk")
            int_prob = 0.5 
            buf_state_value = copy.deepcopy(state_value)
            buf_state_value[target_index] = (1.0 - int_acc) * 0.5 
            _, v, x = self.ego_vehicle_transition(buf_state_value)
            buf_state_value[self.ego_state_index] = x
            buf_state_value[self.ego_state_index+1] = v 
            out_index_list.append([int_prob, self.to_index(buf_state_value)]) 

        else:
            # print("transition if no intervention")
            int_prob = 1.0
            _, v, x = self.ego_vehicle_transition(state_value)
            state_value[self.ego_state_index] = x
            state_value[self.ego_state_index+1] = v 
            out_index_list.append([int_prob, self.to_index(state_value)]) 
            
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
            target_index = int(self.risk_state_index + self.risk_state_len * i)
            if dist > 0.0 and dist < closest_target_dist and state[target_index] >= 0.5:
                closest_target_dist = dist
                closest_target = i

        deceleration_distance = (current_v**2 - self.min_speed**2)/(2*9.8*self.ordinary_G)
        if closest_target is None:
            a = self.ordinary_G*9.8 if current_v < self.ideal_speed else -self.ordinary_G*9.8
        else:
            if closest_target_dist > deceleration_distance:
                a = self.ordinary_G*9.8 if current_v < self.ideal_speed else -self.ordinary_G*9.8
            else:
                a = (self.min_speed**2-current_v**2)/(2*9.8*closest_target_dist)
        # print(a, deceleration_distance, closest_target)
        v = (current_v + a * self.delta_t) * 3.6
        x = current_pose + current_v * self.delta_t + 0.5 * a * self.delta_t**2

        return a, v, x
               

    def to_index(self, state):
        # print("to_index")
        out_index = []
        for i in range(len(state)):
            out_index.append(int((min(max(state[i], self.state_min[i]), self.state_max[i]) - self.state_min[i]) // self.state_width[i]))

        return out_index


    def index_value(self, index, i):
        # print("get index from", index, i)
        return index[i] * self.state_width[i] + self.state_min[i]


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
    dp.init_state_space()
    delta = 1e100
    counter = 0
    try:
        while delta > 0.01:
            delta = dp.value_iteration_sweep()
            counter += 1
            print(counter, delta)
    except Exception as e:
        print(e)

    finally:
        with open("policy.pkl", "wb") as f:
            pickle.dump(dp.policy, f)

        with open("value.pkl", "wb") as f:
            pickle.dump(dp.value_function, f)

        v = dp.value_function[:, :, 1, 3, 1, 2, 1, 1, 2]
        sns.heatmap(v.T, square=False)
        plt.show()

        p = dp.policy[:, :, 1, 3, 1, 2, 1, 1, 2]
        sns.heatmap(p.T, square=False)
        plt.show()

if __name__ == "__main__":
    trial_until_sat()
