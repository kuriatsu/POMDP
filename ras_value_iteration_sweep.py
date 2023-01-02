#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import itertools
import pickle
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import sys

from operator_model import OperatorModel

class MDP:
    def __init__(self, param):

        self.delta_t = param["delta_t"] #1.0
        self.prediction_horizon = param["prediction_horizon"] # 150 
        self.safety_margin = param["safety_margin"] # 5.0
        self.ideal_speed = param["ideal_speed"] # 1.4*10 
        self.min_speed = param["min_speed"] # 2.8 
        self.ordinary_G = param["ordinary_G"] # 0.2
        # self.max_G = 1.0 

        self.p_efficiency = param["p_efficiency"] # -1
        self.p_comfort = param["p_comfort"] # -10
        self.p_ambiguity = param["p_ambiguity"] # -10
        self.p_bad_int_request = param["p_bad_int_request"] # -10
        self.p_int_request = param["p_int_request"] # -1
        self.p_delta_t = param["p_delta_t"] # -1
        self.goal_value = param["goal_value"] # 100

        self.operator_int_prob = param["operator_int_prob"] #0.5
        
        # map 
        self.risk_positions = np.array(param["risk_positions"]).T # [100, 120]

        self.min_step_size = self.prediction_horizon*3.6/self.ideal_speed
        self.discount_factor = self.min_step_size/(self.min_step_size+1.0)

        # ego_pose, ego_vel, 
        self.ego_state_min = np.array([0, 0]).T
        self.ego_state_max = np.array([self.prediction_horizon, self.ideal_speed]).T
        self.ego_state_width = np.array([2, 1.4]).T # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h
        # self.ego_state_width = np.array([1, 0.1]).T # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h

        # risks state: likelihood 0:norisk, 1:risk ,  
        self.risk_state_min = np.array([0.0]*len(self.risk_positions)).T
        self.risk_state_max = np.array([1.0]*len(self.risk_positions)).T
        self.risk_state_width = np.array([0.25]*len(self.risk_positions)).T
        
        # intervention state: int_time, target
        self.int_state_min = np.array([0, -1]).T
        self.int_state_max = np.array([6, len(self.risk_positions)-1]).T
        self.int_state_width = np.array([self.delta_t, 1]).T

        self.state_min = np.r_[self.ego_state_min, self.int_state_min, self.risk_state_min]
        self.state_max = np.r_[self.ego_state_max, self.int_state_max, self.risk_state_max]
        self.state_width = np.r_[self.ego_state_width, self.int_state_width, self.risk_state_width]

        self.index_nums = (1+(self.state_max - self.state_min)/self.state_width).astype(int)
        print("index_nums", self.index_nums)

        self.indexes = None
        self.ego_state_index = 0
        self.int_state_index = len(self.ego_state_width) 
        self.risk_state_index = len(self.ego_state_width) + len(self.int_state_width)
        # 0: not request intervention, else:request intervention
        self.actions = np.arange(-1, len(self.risk_positions)).T
        self.value_function = None
        self.final_state_flag = None
        self.policy = None

        # operator model
        self.operator_model = OperatorModel(
                param["min_time"],
                param["min_time_var"],
                param["acc_time_min"],
                param["acc_time_var"],
                param["acc_time_slope"],
                [i for i in range(int(self.state_min[self.int_state_index]), int(1 + self.state_max[self.int_state_index]), int(self.state_width[self.int_state_index]))],
                )

    def init_state_space(self):

        # indexes
        self.indexes = list(itertools.product(*tuple(range(x) for x in self.index_nums)))
        print("indexes size [byte]", self.indexes.__sizeof__())

        self.value_function, self.final_state_flag = self.init_value_function()
        print("value_function size [byte]", self.value_function.__sizeof__())
        self.policy = self.init_policy()
        print("policy size [byte]", self.policy.__sizeof__())

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
        for prob, index_after in self.state_transition(action, index):
            p_efficiency = 0.0
            p_int_request_penalty = False
            p_ambiguity = 0.0 

            ego_x_before = self.index_value(index, self.ego_state_index)
            ego_v_before = self.index_value(index, self.ego_state_index+1)
            ego_x_after = self.index_value(index_after, self.ego_state_index)
            ego_v_after = self.index_value(index_after, self.ego_state_index+1)

            closest_amb_target = None
            closest_amb = 0.0 
            min_dist = self.prediction_horizon
            for i, risk_position in enumerate(self.risk_positions):
                dist = risk_position - ego_x_before
                risk_prob = self.index_value(index_after, self.risk_state_index+i)
                amb = (0.5 - abs(risk_prob - 0.5))*2
                if 0.0 < dist < min_dist and amb > 0.0:
                    min_dist = dist 
                    closest_amb_target = i
                    closest_amb = amb 
                    
                if ego_x_before <= risk_position <= ego_x_after and ego_v_before > self.min_speed:
                    p_ambiguity = amb 

            if closest_amb_target is not None:
                acceleration_rate = (ego_v_after - ego_v_before) / (self.ordinary_G*9.8)
                p_efficiency = acceleration_rate**2 * closest_amb
                    
            # when change the intervention target, judge the action decision
            # if self.index_value(index, self.int_state_index+1) not in [-1, action]:
            #     int_time = self.index_value(index, self.int_state_index) 
            #     int_acc_list = self.operator_model.get_acc_prob(int_time)
            #     int_acc_list = self.get_int_performance(int_time)
            #     for [int_acc, acc_prob] in :
            #         if int_acc is None:
            #             bad_int_request = acc_prob

            # if intervention after passing the obstacles
            # if self.index_value(index_after, self.ego_state_index) >= self.risk_positions[int(self.index_value(index_after, self.int_state_index+1))]:
            #     bad_int_request = True

            # intervention request penalty
            # if self.index_value(index, self.int_state_index+1) not in [-1, action]:
            if action != -1:
                p_int_request_penalty = True

            action_value = self.p_efficiency*p_efficiency \
                         + self.p_ambiguity*p_ambiguity \
                         + self.p_delta_t*self.delta_t \
                         + self.p_int_request*p_int_request_penalty
                         # + self.p_bad_int_request*bad_int_request \
                         # + self.p_comfort*comfort \
                         # + self.goal_value*self.final_state(index_after)
            # print("action_value", index, action_value, self.p_efficiency*efficiency, self.p_ambiguity*ambiguity, self.p_bad_int_request*bad_int_request)
            value += prob * (self.value_function[tuple(index_after)] + action_value) * self.discount_factor
            # value += prob * (action_value) 
            
        return value

    def state_transition(self, action, index):

        state_value = [self.index_value(index, i) for i in range(len(index))] 
        out_index_list = []

        # intervention state transition 
        if action == self.index_value(index, self.int_state_index+1):
            state_value[self.int_state_index] += self.delta_t
        else:
            state_value[self.int_state_index] = 1
            state_value[self.int_state_index+1] = action

        # print("risk_state and ego_vehicle state") 
        int_time = self.index_value(index, self.int_state_index) 
        int_acc_prob_list = self.operator_model.int_acc(int_time)
        # int_acc_prob_list = self.operator_model.get_acc_prob(int_time)
        target_index = int(self.risk_state_index + self.index_value(index, self.int_state_index+1))

        for [int_acc, acc_prob] in int_acc_prob_list:
            if self.index_value(index, self.int_state_index+1) != action and self.index_value(index, self.int_state_index+1) != -1 and int_acc is not None : 

                # print("transition if target is judged as norisk")
                transition_prob = self.operator_int_prob * acc_prob
                buf_state_value_noint = copy.deepcopy(state_value)
                buf_state_value_noint[target_index] = 1.0 - int_acc 
                _, v, x = self.ego_vehicle_transition(buf_state_value_noint)
                buf_state_value_noint[self.ego_state_index] = x
                buf_state_value_noint[self.ego_state_index+1] = v 
                out_index_list.append([transition_prob, self.to_index(buf_state_value_noint)]) 
                # print("transition if target is judged as risk")
                transition_prob = (1.0 - self.operator_int_prob) * acc_prob
                buf_state_value_int = copy.deepcopy(state_value)
                buf_state_value_int[target_index] = int_acc 
                _, v, x = self.ego_vehicle_transition(buf_state_value_int)
                buf_state_value_int[self.ego_state_index] = x
                buf_state_value_int[self.ego_state_index+1] = v 
                out_index_list.append([transition_prob, self.to_index(buf_state_value_int)]) 

            else:
                # print("transition if no intervention")
                transition_prob = 1.0 * acc_prob
                buf_state_value = copy.deepcopy(state_value)
                _, v, x = self.ego_vehicle_transition(buf_state_value)
                buf_state_value[self.ego_state_index] = x
                buf_state_value[self.ego_state_index+1] = v 
                out_index_list.append([transition_prob, self.to_index(buf_state_value)]) 
            
        # print(out_index_list)
        return out_index_list


    def ego_vehicle_transition(self, state):
        # closest object
        current_v = state[self.ego_state_index+1]
        current_pose = state[self.ego_state_index] 
        acc_list = [] 

        if current_v < self.ideal_speed:
            acc_list.append(self.ordinary_G*9.8)
        elif current_v == self.ideal_speed:
            acc_list.append(0.0)
        else:
            acc_list.append(-self.ordinary_G*9.8)

        # get acceleration value
        for i, pose in enumerate(self.risk_positions):
            target_index = int(self.risk_state_index + i)
            dist = pose - current_pose
            
            if dist < 0.0: # remove target
                is_deceleration_target = False
            elif i == state[self.int_state_index+1]: # intervention target
                is_deceleration_target = state[target_index] >= 0.0
            else:
                is_deceleration_target = state[target_index] >= 0.5

            if not is_deceleration_target:
                continue

            a = 0.0
            deceleration_distance = (current_v**2 - self.min_speed**2)/(2*9.8*self.ordinary_G) + self.safety_margin 
            # keep speed 
            if dist > deceleration_distance+20:
                continue

            # deceleration to the target
            else:
                if dist > self.safety_margin:
                    a = (self.min_speed**2-current_v**2)/(2*(dist-self.safety_margin))
                # if within safety margin, keep speed (expect already min_speed)
                else:
                    # a = (self.min_speed**2-current_v**2)/(2*(dist))
                    a = 0.0

            acc_list.append(a)

        # min speed or max speed
        a = min(acc_list)
        v = current_v + a*self.delta_t
        if v <= self.min_speed:
            v = self.min_speed
            a = 0.0
        elif v >= self.ideal_speed:
            v = self.ideal_speed
            a = 0.0
        x = current_pose + current_v * self.delta_t + 0.5 * a * self.delta_t**2
        # print("a, v, x, current_v, current_x",a, v, x, current_v, current_pose)
        return a, v, x
               

    def to_index(self, state):
        out_index = []
        for i in range(len(state)):
            out_index.append(int((min(max(state[i], self.state_min[i]), self.state_max[i]) - self.state_min[i]) // self.state_width[i]))

        return out_index


    def index_value(self, index, i):
        return index[i] * self.state_width[i] + self.state_min[i]

def trial_until_sat():

    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    
    filename = sys.argv[1].split("/")[-1].split(".")[0]
    dp = MDP(param)
    dp.init_state_space()
    delta = 1e100
    counter = 0
    try:
        while delta > 0.01:
            delta = dp.value_iteration_sweep()
            counter += 1
            print(filename, counter, delta)
    except Exception as e:
        print(e)

    finally:
        with open(f"{filename}_p.pkl", "wb") as f:
            pickle.dump(dp.policy, f)

        with open(f"{filename}_v.pkl", "wb") as f:
            pickle.dump(dp.value_function, f)

            # v = eval("dp.value_function" + param["visualize_elem"])
            # v = dp.value_function[:, :, param["visualize_elems"]]
            # sns.heatmap(v.T, square=False)
            # plt.title(f"{filename}value")
            # plt.savefig(f"{filename}_value.svg")

            # p = eval("dp.policy" + param["visualize_elem"])
            # p = dp.policy[:, :, param["visualize_elems"]]
            # sns.heatmap(p.T, square=False)
            # plt.title(f"{filename}_policy")
            # plt.savefig(f"{filename}_policy.svg")

if __name__ == "__main__":
    trial_until_sat()
