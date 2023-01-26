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
        self.ideal_speed = param["ideal_speed"] # 11.2 
        self.min_speed = param["min_speed"] # 2.8 
        self.ordinary_G = param["ordinary_G"] # 0.2
        
        self.p_omission = param["p_omission"] # -100
        self.p_false_recognition = param["p_false_recognition"] # -1
        self.p_efficiency = param["p_efficiency"] # -1
        self.p_comfort = param["p_comfort"] # -1
        self.p_int_request = param["p_int_request"] # -1

        self.operator_int_prob = param["operator_int_prob"] #0.5

        # MAP
        self.risk_positions = np.array(param["risk_positions"]).T # [100, 120]
        self.discount_factor = 0.95 

        # initial belief
        self.init_ego_pose = param["init_ego_pose"]
        self.init_ego_speed = param["init_ego_speed"]
        self.init_int_target = param["init_int_target"]
        self.init_int_time = param["init_int_time"]
        self.init_risk = param["init_risk"]
        self.init_recognition = param["init_recognition"]

        # POMDPX
        self.root = ET.Element("pomdpx", attrib={"version":"0.1", "id":"cooperative recognition", "xmlns:xsi":"http://www.w3.org/2001/XMLSchema-instance", "xsi:noNamespaceSchemaLocation":"pomdpx.xsd"})
        description = ET.SubElement(self.root, "Description")
        description.text = "This is an auto-generated cooperative recognition POMDPX file"
        discount = ET.SubElement(self.root, "Discount")
        discount.text = "0.95"

        # State Var
        variable = ET.SubElement(self.root, "Variable")
        self.s_ego_pose = np.arange(0, self.prediction_horizon+0.1, 2)
        self.set_state_var(variable, self.s_ego_pose, "ego_pose0", "ego_pose1", "true")

        self.s_ego_speed = np.round(np.arange(self.min_speed, self.ideal_speed+0.1, 1.4), decimals=1) # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h
        self.set_state_var(variable, self.s_ego_speed, "ego_speed0", "ego_speed1", "true")
        
        self.s_int_time = np.round(np.arange(0.0, 10.0+self.delta_t, self.delta_t), decimals=1)
        self.set_state_var(variable, self.s_int_time, "int_time0", "int_time1", "true")

        self.s_int_target = np.arange(-1, len(self.risk_positions))
        self.set_state_var(variable, self.s_int_target, "int_target0", "int_target1", "true")
        
        risk_state = [0, 1]
        for i in range(len(self.risk_positions)):
            self.set_state_var(variable, risk_state, "recognition_"+str(i)+"0", "recognition_"+str(i)+"1", "true")
            self.set_state_var(variable, risk_state, "risk_"+str(i)+"0", "risk_"+str(i)+"1", "false")
        self.s_risk_state = np.array([i for i in itertools.product(risk_state, repeat=len(self.risk_positions))]) 
        self.s_recognition_state = self.s_risk_state.copy() 

        # Action Var
        self.action_int_request =  np.arange(-1, len(self.risk_positions))
        self.set_action_var(variable, "ActionVar", self.action_int_request, "action_int_request")
        self.action_change_recognition =  np.arange(-1, len(self.risk_positions))
        self.set_action_var(variable, "ActionVar", self.action_int_request, "action_change_recognition")
        
        # Obserbation Var
        self.observation = np.array(["int", "no_int", "none"])
        self.set_action_var(variable, "ObsVar", self.observation, "obs_int_behavior")

        # Reward Var
        ET.SubElement(variable, "RewardVar", attrib={"vname":"reward_int_request"})
        ET.SubElement(variable, "RewardVar", attrib={"vname":"reward_safety"})
        ET.SubElement(variable, "RewardVar", attrib={"vname":"reward_comfort"})
        ET.SubElement(variable, "RewardVar", attrib={"vname":"reward_efficiency"})

        # initial state belief
        initial_belief = ET.SubElement(self.root, "InitialStateBelief")
        self.set_initial_state(initial_belief)

        # State Transition Function
        state_tran = ET.SubElement(self.root, "StateTransitionFunction")
        self.set_ego_pose_tran(state_tran)
        self.set_ego_speed_tran(state_tran)
        self.set_int_time_tran(state_tran)
        self.set_risk_tran(state_tran)
        self.set_recognition_tran(state_tran)
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
        self.set_obs(obs_func)

        

    def set_state_var(self, elem, values, prev, curr, obs):
        state = ET.SubElement(elem, "StateVar", attrib={"vnamePrev":prev, "vnameCurr":curr, "fullyObs":obs})
        enum = ET.SubElement(state, "ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_action_var(self, elem, type, values, name):
        state = ET.SubElement(elem, type, attrib={"vname":name})
        enum = ET.SubElement(state, "ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_ego_speed_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "ego_speed1"
        parent = ET.Element("Parent")
        parent.text = "action_int_request ego_pose0 ego_speed0"
        for i in range(len(self.risk_positions)):
            parent.text += " recognition_"+str(i)+"0"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})
        
        for action in self.action_int_request:
            for recognition_state in self.s_recognition_state:
                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{action} - - {' '.join(np.array(recognition_state, dtype='str'))} - " 
                prob_table = ET.Element("ProbTable")
                prob_table.text = ""


                for ego_pose in self.s_ego_pose:
                    for ego_speed in self.s_ego_speed:
                        _, v, _, _ = self.calc_ego_speed(ego_speed, ego_pose, recognition_state, action) 
                        v = min(max(v, min(self.s_ego_speed)), max(self.s_ego_speed))
                        prob_list = np.zeros(len(self.s_ego_speed))
                        prob_list[self.get_index(self.s_ego_speed, v)] = 1.0
                        prob_table.text += " ".join(np.array(prob_list, dtype="str"))
                        prob_table.text += " "
                        
                entry.append(instance)
                entry.append(prob_table)
                parameter.append(entry)
        cond_prob.append(var)
        cond_prob.append(parent)
        cond_prob.append(parameter)


    def get_index(self, list, value):
        min_d = 1e1000 
        min_index = None
        for i, l in enumerate(list):
            d = abs(l-value)
            if d < min_d:
                min_d = d
                min_index = i
        return min_index

    def set_ego_pose_tran(self, elem):

        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "ego_pose1"
        parent = ET.Element("Parent")
        parent.text = "action_int_request ego_pose0 ego_speed0"
        for i in range(len(self.risk_positions)):
            parent.text += " recognition_"+str(i)+"0"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})

        for action in self.action_int_request:
            for recognition_state in self.s_recognition_state:

                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{action} - - {' '.join(np.array(recognition_state, dtype='str'))} - " 
                prob_table = ET.Element("ProbTable")
                prob_table.text = ""

                for ego_pose in self.s_ego_pose:
                    for ego_speed in self.s_ego_speed:
                        _, _, x, _ = self.calc_ego_speed(ego_speed, ego_pose, recognition_state, action) 
                        x = min(max(x, min(self.s_ego_pose)), max(self.s_ego_pose))
                        prob_list = np.zeros(len(self.s_ego_pose))
                        prob_list[self.get_index(self.s_ego_pose, x)] = 1.0
                        prob_table.text += " ".join(np.array(prob_list, dtype="str"))
                        prob_table.text += " "

                entry.append(instance)
                entry.append(prob_table)
                parameter.append(entry)
        cond_prob.append(var)
        cond_prob.append(parent)
        cond_prob.append(parameter)
    
    
    def calc_ego_speed(self, ego_speed, ego_pose, recognition_state, action):
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
            dist = pose - ego_pose
            
            # find deceleration target
            if dist < 0.0: # passed target
                is_deceleration_target = False
            elif i == action: # intervention target
                is_deceleration_target = recognition_state[i] == 1.0 
            else: # not intervention target
                is_deceleration_target = recognition_state[i] == 1.0

            if not is_deceleration_target:
                continue

            a = 0.0
            deceleration_distance = (ego_speed**2 - self.min_speed**2)/(2*9.8*self.ordinary_G) + self.safety_margin 
            # keep speed, 20=discretization eps
            if dist > deceleration_distance+2:
            # if dist > deceleration_distance:
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
        decel_target = acc_list.index(a)
        v = ego_speed + a*self.delta_t
        if v <= self.min_speed:
            v = self.min_speed
            a = 0.0
        elif v >= self.ideal_speed:
            v = self.ideal_speed
            a = 0.0

        x = ego_pose + ego_speed * self.delta_t + 0.5 * a * self.delta_t**2
        # x = ego_pose + ego_speed * self.delta_t
        # x = int((x-min(self.s_ego_pose))//(self.s_ego_pose[1]-self.s_ego_pose[0]))
        # v = int((v-min(self.s_ego_speed))//(self.s_ego_speed[1]-self.s_ego_speed[0]))
        return a, v, x, decel_target


    def set_risk_tran(self, elem):
        for i in range(len(self.risk_positions)):
            cond_prob = ET.SubElement(elem, "CondProb")
            var = ET.Element("Var")
            var.text = "risk_"+str(i)+"1"
            parent = ET.Element("Parent")
            parent.text = "risk_"+str(i)+"0"
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
            

    def set_recognition_tran(self, elem):
        for i in range(len(self.risk_positions)):
            cond_prob = ET.SubElement(elem, "CondProb")
            var = ET.SubElement(cond_prob, "Var")
            var.text = "recognition_"+str(i)+"1"
            parent = ET.SubElement(cond_prob, "Parent")
            parent.text = "action_change_recognition action_int_request recognition_"+str(i)+"0"
            parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
            
            for a_c, a_i in itertools.product(self.action_change_recognition, self.action_int_request):
                if a_i == i:
                    entry = ET.SubElement(parameter, "Entry")
                    instance = ET.SubElement(entry, "Instance")
                    instance.text = f"{a_c} {a_i} * 1 " 
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = "1.0"
                
                elif a_c == i:

                    entry = ET.SubElement(parameter, "Entry")
                    instance = ET.SubElement(entry, "Instance")
                    instance.text = f" {a_c} {a_i} 1 0 " 
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = "1.0"

                    entry = ET.SubElement(parameter, "Entry")
                    instance = ET.SubElement(entry, "Instance")
                    instance.text = f" {a_c} {a_i} 0 1 " 
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = "1.0"

                else:
                    entry = ET.SubElement(parameter, "Entry")
                    instance = ET.SubElement(entry, "Instance")
                    instance.text = f" {a_c} {a_i} - - " 
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = "identity"

            


    def set_int_time_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "int_time1"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "action_int_request int_target0 int_time0"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})

        entry = ET.Element("Entry")
        instance = ET.Element("Instance")
        instance.text = f"-1 * * 0.0" 
        prob_table = ET.Element("ProbTable")
        prob_table.text = "1.0"
        entry.append(instance)
        entry.append(prob_table)
        parameter.append(entry)

        for a in self.action_int_request:
            if a == -1: continue
            for t in self.s_int_target:
                entry = ET.Element("Entry")
                instance = ET.SubElement(entry, "Instance")
                if a != t:
                    instance.text = f"{a} {t} * 1.0 " 
                    prob_list = []
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = "1.0"
                else:
                    instance.text = f"{a} {t} - - " 
                    prob_list = []
                    for time in self.s_int_time:
                        buf = [0.0]*len(self.s_int_time)
                        buf[int(min(time+1, max(self.s_int_time))//(self.s_int_time[1]-self.s_int_time[0]))] = 1.0
                        prob_list += buf 
                    prob_table = ET.SubElement(entry, "ProbTable")
                    prob_table.text = " ".join([str(_) for _ in prob_list])

                parameter.append(entry)


    def set_target_tran(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "int_target1"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "action_int_request int_target0"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})

        for a in self.action_int_request:
            entry = ET.Element("Entry")
            instance = ET.SubElement(entry, "Instance")
            instance.text = f" {a} * {a} " 
            prob_table = ET.SubElement(entry, "ProbTable")
            prob_table.text = "1.0"
            
            parameter.append(entry)


    def set_reward(self, elem):

        # driving safety
        func_safety = ET.SubElement(elem, "Func")
        var_safety = ET.SubElement(func_safety, "Var")
        var_safety.text = "reward_safety"
        parent_safety = ET.SubElement(func_safety, "Parent")
        parent_safety.text = f"ego_pose0 ego_pose1 " + " ".join([f"recognition_{i}0 " for i in range(len(self.risk_positions))])  + " ".join([f"risk_{i}0 " for i in range(len(self.risk_positions))])
        parameter_safety = ET.SubElement(func_safety, "Parameter", attrib={"type":"TBL"})

        ##  driving efficiency
        func_eff = ET.SubElement(elem, "Func")
        var_eff = ET.SubElement(func_eff, "Var")
        var_eff.text = "reward_efficiency"
        parent_eff = ET.SubElement(func_eff, "Parent")
        parent_eff.text = f"ego_pose0 ego_pose1 ego_speed0 " + " ".join([f"recognition_{i}0 " for i in range(len(self.risk_positions))])
        parameter_eff = ET.SubElement(func_eff, "Parameter", attrib={"type":"TBL"})

        for pose_prev in self.s_ego_pose:
            for speed_prev in self.s_ego_speed:
                for recognition_state in self.s_recognition_state:
                    _, speed_curr, pose_curr, target = self.calc_ego_speed(speed_prev, pose_prev, recognition_state, -1) 
                    speed_curr = min(max(speed_curr, min(self.s_ego_speed)), max(self.s_ego_speed))
                    speed_curr = self.s_ego_speed[self.get_index(self.s_ego_speed, speed_curr)]
                    pose_curr = min(max(pose_curr, min(self.s_ego_pose)), max(self.s_ego_pose))
                    pose_curr = self.s_ego_pose[self.get_index(self.s_ego_pose, pose_curr)]
                    for i, risk_position in enumerate(self.risk_positions):
                        if pose_prev <= risk_position < pose_curr: 

                            # driving safety
                            entry_safety = ET.SubElement(parameter_safety, "Entry")
                            instance_safety = ET.SubElement(entry_safety, "Instance")
                            prob_table_safety = ET.SubElement(entry_safety, "ValueTable")

                            risk_state = ["*"]*len(self.risk_positions)
                            if recognition_state[i] == 0:
                                risk_state[i] = "1"    
                                prob_table_safety.text = str(self.p_omission) 

                            else:
                                risk_state[i] = "0"
                                prob_table_safety.text = str(self.p_false_recognition) 

                            instance_safety.text = f"{pose_prev} {pose_curr} {' '.join([str(r) for r in recognition_state])} {' '.join(risk_state)}"

                            # driving efficiency
                            entry_eff = ET.SubElement(parameter_eff, "Entry")
                            instance_eff = ET.SubElement(entry_eff, "Instance")
                            prob_table_eff = ET.SubElement(entry_eff, "ValueTable")
                            instance_eff.text = f"{pose_prev} {pose_curr} {speed_prev} {' '.join([str(r) for r in recognition_state])}"
                            if recognition_state[i] == 0:
                                value = (max(self.s_ego_speed) - speed_prev) / (max(self.s_ego_speed) - min(self.s_ego_speed))
                            else:
                                value = (speed_prev - min(self.s_ego_speed)) / (max(self.s_ego_speed) - min(self.s_ego_speed))
                            prob_table_eff.text = str(value*self.p_efficiency)

        # driving comfort
        func = ET.SubElement(elem, "Func")
        var = ET.SubElement(func, "Var")
        var.text = "reward_comfort"
        parent = ET.SubElement(func, "Parent")
        parent.text = f"ego_speed0 ego_speed1"
        parameter = ET.SubElement(func, "Parameter", attrib={"type":"TBL"})

        for speed_prev in self.s_ego_speed:
            for speed_curr in self.s_ego_speed:
                entry = ET.SubElement(parameter, "Entry")

                instance = ET.SubElement(entry, "Instance")
                instance.text = f"{speed_prev} {speed_curr}"

                prob_table = ET.SubElement(entry, "ValueTable")
                value = ((speed_curr - speed_prev)/(max(self.s_ego_speed)-min(self.s_ego_speed)))**2
                prob_table.text = str(value*self.p_comfort) 


        # intervention request
        func = ET.SubElement(elem, "Func")
        var = ET.SubElement(func, "Var")
        var.text = "reward_int_request"
        parent = ET.SubElement(func, "Parent")
        parent.text = "action_int_request"
        parameter = ET.SubElement(func, "Parameter", attrib={"type":"TBL"})
        for a in self.action_int_request:
            if a == -1: continue
            entry = ET.Element("Entry")
            instance = ET.Element("Instance")
            instance.text = f"{a}"
            prob_table = ET.Element("ValueTable")
            prob_table.text = f"{self.p_int_request}"
            entry.append(instance)
            entry.append(prob_table)
            parameter.append(entry)


    def set_obs(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "obs_int_behavior"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "action_int_request int_time1 " + " ".join([f"risk_{_}1" for _ in range(len(self.risk_positions))])
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})

        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = f"* * " + "* "*len(self.risk_positions) + " none "
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_table.text = "1.0"
        
        for int_time in self.s_int_time:
            acc = self.operator_model.int_acc(int_time)
            if acc is None: continue
            for a in self.action_int_request:
                for risks in itertools.product([0, 1], repeat=len(self.risk_positions)):
                    entry = ET.SubElement(parameter, "Entry")
                    instance = ET.SubElement(entry, "Instance")
                    instance.text = f"{a} {int_time} " + " ".join([str(_) for _ in risks]) + " - "
                    prob_table = ET.SubElement(entry, "ProbTable")
                    if risks[a] == 1:
                        prob_table.text = f"{1.0 - acc} {acc} 0.0"
                    elif risks[a] == 0:
                        prob_table.text = f"{acc} {1.0 - acc} 0.0"


    def set_obs_bak(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "obs_int_behavior"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "action_int_request int_time1 int_target1 " + " ".join([f"risk_{_}1" for _ in range(len(self.risk_positions))])
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})

        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = f"* * * " + "* "*len(self.risk_positions) + "none"
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_table.text = "1.0"
        
        for int_time in self.s_int_time:
            acc = self.operator_model.int_acc(int_time)
            if acc is None: continue
            for a in self.action_int_request:
                for int_target in self.s_int_target:
                    if int_target in [-1, a]: continue
                    for risks in itertools.product([0, 1], repeat=len(self.risk_positions)):
                        entry = ET.SubElement(parameter, "Entry")
                        instance = ET.SubElement(entry, "Instance")
                        instance.text = f"{a} {int_time} {int_target} " + " ".join([str(_) for _ in risks]) + " - "
                        acc = self.operator_model.int_acc(int_time)
                        prob_table = ET.SubElement(entry, "ProbTable")
                        if risks[int_target] == 1:
                            prob_table.text = f"{1.0 - acc} {acc} 0.0"
                        elif risks[int_target] == 0:
                            prob_table.text = f"{acc} {1.0 - acc} 0.0"


    def set_initial_state(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "ego_pose0"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "null"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = " - "
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_list = [0.0]*len(self.s_ego_pose)
        prob_list[np.where(self.s_ego_pose==self.init_ego_pose)[0][0]] = 1.0
        prob_table.text = " ".join([str(_) for _ in prob_list]) 

        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "ego_speed0"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "null"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = " - "
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_list = [0.0]*len(self.s_ego_speed)
        prob_list[np.where(self.s_ego_speed==self.init_ego_speed)[0][0]] = 1.0
        prob_table.text = " ".join([str(_) for _ in prob_list]) 

        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "int_time0"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "null"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = " - "
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_list = [0.0]*len(self.s_int_time)
        prob_list[np.where(self.s_int_time==self.init_int_time)[0][0]] = 1.0
        prob_table.text = " ".join([str(_) for _ in prob_list]) 
        
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.SubElement(cond_prob, "Var")
        var.text = "int_target0"
        parent = ET.SubElement(cond_prob, "Parent")
        parent.text = "null"
        parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
        entry = ET.SubElement(parameter, "Entry")
        instance = ET.SubElement(entry, "Instance")
        instance.text = " - "
        prob_table = ET.SubElement(entry, "ProbTable")
        prob_list = [0.0]*len(self.s_int_target)
        prob_list[np.where(self.s_int_target==self.init_int_target)[0][0]] = 1.0
        prob_table.text = " ".join([str(_) for _ in prob_list]) 

        for i in range(len(self.risk_positions)):
            cond_prob = ET.SubElement(elem, "CondProb")
            var = ET.SubElement(cond_prob, "Var")
            var.text = "risk_"+str(i)+"0"
            parent = ET.SubElement(cond_prob, "Parent")
            parent.text = "null"
            parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
            entry = ET.SubElement(parameter, "Entry")
            instance = ET.SubElement(entry, "Instance")
            instance.text = " - "
            prob_table = ET.SubElement(entry, "ProbTable")
            risk = self.init_risk[i]
            prob_table.text = f"{1.0-risk} {risk}"

        for i in range(len(self.risk_positions)):
            cond_prob = ET.SubElement(elem, "CondProb")
            var = ET.SubElement(cond_prob, "Var")
            var.text = "recognition_"+str(i)+"0"
            parent = ET.SubElement(cond_prob, "Parent")
            parent.text = "null"
            parameter = ET.SubElement(cond_prob, "Parameter", attrib={"type":"TBL"})
            entry = ET.SubElement(parameter, "Entry")
            instance = ET.SubElement(entry, "Instance")
            instance.text = " - "
            prob_table = ET.SubElement(entry, "ProbTable")
            recognition = self.init_recognition[i]
            prob_table.text = f"{1-recognition} {recognition}"

def trial_until_sat():

    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    
    dp = MDP(param)
    tree = ET.ElementTree(element=dp.root)
    ET.indent(tree, space="    ", level=0)
    filename = sys.argv[1].split(".")[0]
    tree.write(filename+".pomdpx", encoding="UTF-8", xml_declaration=True)
    
if __name__ == "__main__":
    trial_until_sat()
