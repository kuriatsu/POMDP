#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import yaml
import sys
import xml.etree.ElementTree as ET
import itertools

from operator_model import OperatorModel

class MDP:
    def __init__(self, param):

        self.delta_t = param["delta_t"] #1.0
        self.prediction_horizon = param["prediction_horizon"] # 150 
        self.safety_margin = param["safety_margin"] # 5.0
        self.ideal_speed = param["ideal_speed"] # 1.4*10 
        self.min_speed = param["min_speed"] # 2.8 
        self.ordinary_G = param["ordinary_G"] # 0.2
        
        self.p_efficiency = param["p_efficiency"] # -1
        self.p_ambiguity = param["p_ambiguity"] # -10
        self.p_int_request = param["p_int_request"] # -1

        self.operator_int_prob = param["operator_int_prob"] #0.5

        # MAP
        self.risk_positions = np.array(param["risk_positions"]).T # [100, 120]
        self.discount_factor = 0.95 


        # POMDPX
        self.root = ET.Element("pomdpx", attrib={"version":"0.1", "id":"cooperative recognition", "xmlns":"", "xsi":""})
        ET.SubElement(self.root, "Description")
        discount = ET.SubElement(self.root, "Discount")
        discount.text = 0.95

        # State Var
        variable = ET.SubElement(self.root, "Variable")
        self.s_ego_pose = np.arange(0.0, self.prediction_horizon, 2.0).T
        self.set_state_var(variable, "StateVar", self.s_ego_pose, "ego_pose0", "ego_pose1", "true")

        self.s_ego_speed = np.arange(self.min_speed, self.ideal_speed, 1.4).T # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h
        self.set_state_var(variable, self.s_ego_speed, "ego_speed0", "ego_speed1", "true")
        
        self.s_int_time = np.arange(0.0, 10.0, self.delta_t).T
        self.set_state_var(variable, "StateVar", self.s_int_time, "int_time0", "int_time1", "true")

        self.s_int_target = np.arange(-1, len(self.risk_positions)).T
        self.set_state_var(variable, "StateVar", self.s_int_target, "int_target0", "int_target1", "true")

        s_risk_state = np.array([0, 1])
        self.s_risk_state = np.array(itertools.product(s_risk_state, repeat=len(self.risk_positions))) 
        for i in range(len(self.risk_positions)):
            self.set_state_var(variable, "StateVar", s_risk_state, "risk_"+i+"0", "risk_"+i+"1", "false")

        # Action Var
        self.actions = np.arange(-1, len(self.risk_positions)).T
        self.set_action_var(variable, "ActionVar", self.actions, "action_int_request")
        
        # Obserbation Var
        self.observation = np.array(["no_int", "int"]).T
        self.set_action_var(variable, "ObsVar", self.observation, "obs_int_behavior")

        # State Transition Function
        state_tran = ET.SubElement(self.root, "StateTransitionFunction")
        self.set_ego_speed_tran(state_tran)
        self.set_ego_pose_tran(state_tran)
        self.set_int_time_tran(state_tran)
        self.set_risk_tran(state_tran)
        self.set_target_tran(state_tran)

        # Reward
        reward_func = ET.SubElement(self.root, "RewardFunction")
        self.set_reward(reward_func)

        # Observation
        self.operator_model = OperatorModel(
                param["min_time"],
                param["min_time_var"],
                param["acc_time_min"],
                param["acc_time_var"],
                param["acc_time_slope"],
                self.s_int_time,
                )
        obs_func = ET.SubElement(self.root, "ObsFunction")

        initial_belief = ET.SubElement(self.root, "InitialStateBelief")

    def set_state_var(self, elem, values, prev, curr, obs):
        state = elem.SubElement("StateVar", attrib={"vnamePrev":prev, "vnameCurr":curr, "fullyObs":obs})
        enum = state.SubElement("ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_action_var(self, elem, type, values, name):
        state = elem.SubElement(type, attrib={"vname":name})
        enum = state.SubElement("ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_ego_speed_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "ego_speed1"
        parent = ET.Element("Parent")
        parent.text = "action_int_request ego_pose0 ego_speed0"
        for i in range(len(self.risk_positions)):
            parent.text += " risk_"+i+"0"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})

        for a in self.actions:
            for risk_state in self.s_risk_state:
                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{a} - - {' '.join(np.array(risk_state, dtype='str'))} - " 
                prob_table = ET.Element("ProbTable")
                prob_table.text = ""
                for ego_pose in self.s_ego_pose:
                    for ego_speed in self.s_ego_speed:
                        v = self.calc_ego_speed(ego_speed, ego_pose, risk_state, a) 
                        prob_list = np.zeros(len(self.s_ego_speed))
                        prob_list[v//(self.s_ego_speed[1]-self.s_ego_speed[0])] = 1.0
                        prob_table.text += " ".join(np.array(prob_list, dtype="str"))
                        prob_table.text += " "
                        
                entry.append(instance)
                entry.append(prob_table)
            parameter.append(entry)
        cond_prob.append(var)
        cond_prob.append(parent)
        cond_prob.append(parameter)

    
    def calc_ego_speed(self, ego_speed, ego_pose, risk_state, action):
        """move ego vehile (mainly calcurate acceleration)
        state : state value (not index)
        """
        acc_list = [] 

        # acceleration to keep ideal speed 
        if ego_speed < self.ideal_speed:
            acc_list.append(self.ordinary_G*9.8)
        elif ego_speed == self.ideal_speed:
            acc_list.append(0.0)
        else:
            acc_list.append(-self.ordinary_G*9.8)

        # get deceleration value against target
        for i, pose in enumerate(self.risk_positions):
            target_index = int(self.risk_state_index + i)
            dist = pose - ego_pose
            
            # find deceleration target
            if dist < 0.0: # passed target
                is_deceleration_target = False
            elif i == action: # intervention target
                is_deceleration_target = risk_state[i] == 1.0 
            else: # not intervention target
                is_deceleration_target = risk_state[i] == 1.0

            if not is_deceleration_target:
                continue

            a = 0.0
            deceleration_distance = (ego_speed**2 - self.min_speed**2)/(2*9.8*self.ordinary_G) + self.safety_margin 
            # keep speed, 20=discretization eps
            # if dist > deceleration_distance+20:
            if dist > deceleration_distance:
                continue

            # deceleration to the target
            else:
                if dist > self.safety_margin:
                    a = (self.min_speed**2-ego_speed**2)/(2*(dist-self.safety_margin))
                # if within safety margin, keep speed (expect already min_speed)
                else:
                    # a = (self.min_speed**2-ego_speed**2)/(2*(dist))
                    a = 0.0

            acc_list.append(a)

        # clip to min speed or max speed
        a = min(acc_list)
        v = ego_speed + a*self.delta_t
        if v <= self.min_speed:
            v = self.min_speed
            a = 0.0
        elif v >= self.ideal_speed:
            v = self.ideal_speed
            a = 0.0
        return v 


    def set_ego_pose_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "ego_pose1"
        parent = ET.Element("Parent")
        parent.text = "ego_pose0 ego_speed0"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})

        entry = ET.Element("Entry")
        instance = ET.Element("Instance")
        instance.text = f" - - - " 
        prob_table = ET.Element("ProbTable")
        prob_table.text = ""
        for ego_pose in self.s_ego_pose:
            for ego_speed in self.s_ego_speed:
                x = ego_pose + ego_speed * self.delta_t + 0.5 * a * self.delta_t**2
                prob_list = np.zeros(len(self.s_ego_pose))
                prob_list[x//(self.s_ego_pose[1]-self.s_ego_pose[0])] = 1.0
                prob_table.text += " ".join(np.array(prob_list, dtype="str"))
                prob_table.text += " "
                
        entry.append(instance)
        entry.append(prob_table)
        parameter.append(entry)
        cond_prob.append(var)
        cond_prob.append(parent)
        cond_prob.append(parameter)
               
    def set_risk_tran(self, elem):
        for i in range(len(self.risk_positions)):
            cond_prob = ET.SubElement(elem, "CondProb")
            var = ET.Element("Var")
            var.text = "risk_"+i+"1"
            parent = ET.Element("Parent")
            parent.text = "risk_"+i+"0"
            parameter = ET.Element("Parameter", attrib={"type":"TBL"})

            entry = ET.Element("Entry")
            instance = ET.Element("Instance")
            instance.text = f" - - " 
            prob_table = ET.Element("ProbTable")
            prob_table.text = "identity"
                
            entry.append(instance)
            entry.append(prob_table)
            parameter.append(entry)
            cond_prob.append(var)
            cond_prob.append(parent)
            cond_prob.append(parameter)
            
        
    def set_int_time_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "int_time1"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "action_int_request int_target1 int_time0"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})

        entry = ET.Element("Entry")
        instance = ET.Element("Instance")
        instance.text = f"-1 * * 0" 
        prob_table = ET.Element("ProbTable")
        prob_table.text = "1.0"
        entry.append(instance)
        entry.append(prob_table)
        parameter.append(entry)

        for a in len(self.risk_positions):
            for t in self.action_int_request:
                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{a} {t} - -" 
                prob_list = []
                for time in self.s_int_time:
                    buf = [0]*len(self.s_int_time)
                    buf[max(time+1, self.s_int_time[-1])//(self.s_int_time[1]-self.s_int_time[0])] = 1.0
                    prob_list.append(buf) 
                prob_table = ET.Element("ProbTable")
                prob_table.text = " ".join(np.arange(prob_list, dtype="str"))

                entry.append(instance)
                entry.append(prob_table)
                parameter.append(entry)


    def set_target_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "int_target1"
        parent = ET.Element("Parent")
        parent.text = "action_int_request"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})

        entry = ET.Element("Entry")
        instance = ET.Element("Instance")
        instance.text = f" - - " 
        prob_table = ET.Element("ProbTable")
        prob_table.text = "identity"
            
        entry.append(instance)
        entry.append(prob_table)
        parameter.append(entry)
        cond_prob.append(var)
        cond_prob.append(parent)
        cond_prob.append(parameter)


    def final_state(self, index):
        return index[0] == float(self.index_nums[0]) - 1.0

    def value_iteration_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flag[index]:
                for action in self.actions:
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

                        if action != -1:
                            p_int_request_penalty = True

                        action_value = self.p_efficiency*p_efficiency \
                                     + self.p_ambiguity*p_ambiguity \
                                     + self.p_delta_t*self.delta_t \
                                     + self.p_int_request*p_int_request_penalty
            
                        if self.index_value(index, self.int_state_index+1) not in [action, -1]:

                            int_prob_list = self.operator_model.int_prob(self.index_value(index, self.int_state_index))
                        transition = f"T : {action} : {index} : {index_after} : {self.operator_int_prob}"
                        reward = f"R : {action} : {index} : {index_after} : {action_value}"
                        observation = f"O : {action} : {index_after} : risk : {prob}"
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
            if self.index_value(index, self.int_state_index+1) not in [action, -1] and int_acc is not None : 

                # print("transition if target is judged as norisk")
                transition_prob = self.operator_int_prob * acc_prob
                buf_state_value_noint = copy.deepcopy(state_value)
                buf_state_value_noint[target_index] = 0.0 
                _, v, x = self.ego_vehicle_transition(buf_state_value_noint)
                buf_state_value_noint[self.ego_state_index] = x
                buf_state_value_noint[self.ego_state_index+1] = v 
                out_index_list.append([transition_prob, self.to_index(buf_state_value_noint)]) 

                # print("transition if target is judged as risk")
                transition_prob = (1.0 - self.operator_int_prob) * acc_prob
                buf_state_value_int = copy.deepcopy(state_value)
                buf_state_value_int[target_index] = 1.0 
                _, v, x = self.ego_vehicle_transition(buf_state_value_int)
                buf_state_value_int[self.ego_state_index] = x
                buf_state_value_int[self.ego_state_index+1] = v 
                out_index_list.append([transition_prob, self.to_index(buf_state_value_int)]) 

        return out_index_list

    def to_index(self, state):
        """ get index from state value
        """
        out_index = []
        for i in range(len(state)):
            out_index.append(int((min(max(state[i], self.state_min[i]), self.state_max[i]) - self.state_min[i]) // self.state_width[i]))

        return out_index


    def index_value(self, index, i):
        """ get state value from index
        """
        return index[i] * self.state_width[i] + self.state_min[i]

def trial_until_sat():

    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    
    filename = sys.argv[1].split(".")[]
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

if __name__ == "__main__":
    trial_until_sat()
