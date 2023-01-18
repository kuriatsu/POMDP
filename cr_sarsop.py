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
        self.p_int_request = param["p_int_request"] # -1

        self.operator_int_prob = param["operator_int_prob"] #0.5

        # MAP
        self.risk_positions = np.array(param["risk_positions"]).T # [100, 120]
        self.discount_factor = 0.95 


        # POMDPX
        self.root = ET.Element("pomdpx", attrib={"version":"0.1", "id":"cooperative recognition", "xmlns":"", "xsi":""})
        ET.SubElement(self.root, "Description")
        discount = ET.SubElement(self.root, "Discount")
        discount.text = "0.95"

        # State Var
        variable = ET.SubElement(self.root, "Variable")
        self.s_ego_pose = np.arange(0.0, self.prediction_horizon, 2.0)
        np.append(self.s_ego_pose, self.s_ego_pose[-1]+2.0)
        self.set_state_var(variable, self.s_ego_pose, "ego_pose0", "ego_pose1", "true")

        self.s_ego_speed = np.arange(self.min_speed, self.ideal_speed, 1.4) # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h
        np.append(self.s_ego_speed, self.s_ego_speed[-1]+1.4) # min speed=10km/h=2.8m/s, delta_t=1.0s, 1.4=5km/h
        self.set_state_var(variable, self.s_ego_speed, "ego_speed0", "ego_speed1", "true")
        
        self.s_int_time = np.arange(0.0, 10.0, self.delta_t)
        np.append(self.s_int_time, self.s_int_time[-1]+self.delta_t)
        self.set_state_var(variable, self.s_int_time, "int_time0", "int_time1", "true")

        self.s_int_target = np.arange(-1, len(self.risk_positions)).T
        self.set_state_var(variable, self.s_int_target, "int_target0", "int_target1", "true")
        
        risk_state = [0, 1]
        self.s_risk_state = np.array([i for i in itertools.product(risk_state, repeat=len(self.risk_positions))]) 
        print(self.s_risk_state)
        for i in range(len(self.risk_positions)):
            self.set_state_var(variable, risk_state, "risk_"+str(i)+"0", "risk_"+str(i)+"1", "false")

        # Action Var
        self.action_int_request =  np.arange(-1, len(self.risk_positions)).T
        self.set_action_var(variable, "ActionVar", self.action_int_request, "action_int_request")
        
        # Obserbation Var
        self.observation = np.array(["no_int", "int"]).T
        self.set_action_var(variable, "ObsVar", self.observation, "obs_int_behavior")

        # State Transition Function
        state_tran = ET.SubElement(self.root, "StateTransitionFunction")
        self.set_ego_tran(state_tran)
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
        self.set_obs(obs_func)

        initial_belief = ET.SubElement(self.root, "InitialStateBelief")
        self.set_initial_state(initial_belief)
        

    def set_state_var(self, elem, values, prev, curr, obs):
        state = ET.SubElement(elem, "StateVar", attrib={"vnamePrev":prev, "vnameCurr":curr, "fullyObs":obs})
        enum = ET.SubElement(state, "ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_action_var(self, elem, type, values, name):
        state = ET.SubElement(elem, type, attrib={"vname":name})
        enum = ET.SubElement(state, "ValueEnum")
        enum.text = " ".join(np.array(values, dtype="str"))

    def set_ego_tran(self, elem):
        cond_prob_v = ET.SubElement(elem, "CondProb")
        var_v = ET.Element("Var")
        var_v.text = "ego_speed1"
        parent_v = ET.Element("Parent")
        parent_v.text = "action_int_request ego_pose0 ego_speed0"
        for i in range(len(self.risk_positions)):
            parent_v.text += " risk_"+str(i)+"0"
        parameter_v = ET.Element("Parameter", attrib={"type":"TBL"})

        cond_prob_x = ET.SubElement(elem, "CondProb")
        var_x = ET.Element("Var")
        var_x.text = "ego_pose1"
        parent_x = ET.Element("Parent")
        parent_x.text = "action_int_request ego_pose0 ego_speed0"
        for i in range(len(self.risk_positions)):
            parent_x.text += " risk_"+str(i)+"0"
        parameter_x = ET.Element("Parameter", attrib={"type":"TBL"})

        
        for a in self.action_int_request:
            for risk_state in self.s_risk_state:
                entry_v = ET.Element("Entry")
                instance_v = ET.Element("Instance")
                instance_v.text = f"{a} - - {' '.join(np.array(risk_state, dtype='str'))} - " 
                prob_table_v = ET.Element("ProbTable")
                prob_table_v.text = ""

                entry_x = ET.Element("Entry")
                instance_x = ET.Element("Instance")
                instance_x.text = f"{a} - - {' '.join(np.array(risk_state, dtype='str'))} - " 
                prob_table_x = ET.Element("ProbTable")
                prob_table_x.text = ""

                for ego_pose in self.s_ego_pose:
                    for ego_speed in self.s_ego_speed:
                        a, v = self.calc_ego_speed(ego_speed, ego_pose, risk_state, a) 
                        v = min(max(v, min(self.s_ego_speed)), max(self.s_ego_speed))
                        prob_list_v = np.zeros(len(self.s_ego_speed))
                        prob_list_v[int((v-self.s_ego_speed[0])//(self.s_ego_speed[1]-self.s_ego_speed[0]))] = 1.0
                        prob_table_v.text += " ".join(np.array(prob_list_v, dtype="str"))
                        prob_table_v.text += " "
                        
                        # x = ego_pose + ego_speed * self.delta_t + 0.5 * a * self.delta_t**2
                        x = ego_pose + ego_speed * self.delta_t
                        x = min(max(x, min(self.s_ego_pose)), max(self.s_ego_pose))
                        prob_list_x = np.zeros(len(self.s_ego_pose))
                        prob_list_x[int((x-self.s_ego_pose[0])//(self.s_ego_pose[1]-self.s_ego_pose[0]))] = 1.0
                        prob_table_x.text += " ".join(np.array(prob_list_x, dtype="str"))
                        prob_table_x.text += " "

                entry_x.append(instance_x)
                entry_v.append(instance_v)
                entry_x.append(prob_table_x)
                entry_v.append(prob_table_v)

            parameter_x.append(entry_x)
            parameter_v.append(entry_v)

        cond_prob_x.append(var_x)
        cond_prob_x.append(parent_x)
        cond_prob_x.append(parameter_x)
        cond_prob_v.append(var_v)
        cond_prob_v.append(parent_v)
        cond_prob_v.append(parameter_v)

    
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
        return a, v 


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

        for a in range(len(self.risk_positions)):
            for t in self.action_int_request:
                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{a} {t} - -" 
                prob_list = []
                for time in self.s_int_time:
                    buf = [0.0]*len(self.s_int_time)
                    buf[int(min(time+1, max(self.s_int_time))//(self.s_int_time[1]-self.s_int_time[0]))] = 1.0
                    prob_list += buf 
                prob_table = ET.Element("ProbTable")
                prob_table.text = " ".join([str(_) for _ in prob_list])

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

    def set_reward(self, elem):
        func = ET.SubElement(elem, "Func")
        var = ET.Element("Var")
        var.text = "reward"
        parent = ET.Element("Parent")
        parent.text = "ego_pose0 ego_speed0" + " ".join([f"risk_{_}0" for _ in range(len(self.risk_positions))])
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})

        # driving comfort
        for risks in itertools.product([0, 1], repeat=len(self.risk_positions)):
            for idx, ego_position in enumerate(self.risk_positions):
                entry = ET.Element("Entry")
                instance = ET.Element("Instance")
                instance.text = f"{ego_position} - " + " ".join([str(_) for _ in risks])
                prob_table = ET.Element("ProbTable")
                values = []
                for speed in self.s_ego_speed:
                    values.append(speed/max(self.s_ego_speed) * risks[idx])
                prob_table.text = " ".join([str(_) for _ in values])

                entry.append(instance)
                entry.append(prob_table)
                parameter.append(entry)
        func.append(var)
        func.append(parent)
        func.append(parameter)
        elem.append(func)

        # intervention request
        func = ET.Element(elem)
        var = ET.Element("Var")
        var.text = "reward_int_request"
        parent = ET.Element("Parent")
        parent.text = "action_int_request"
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})
        for a in self.action_int_request:
            if a == -1: continue
            entry = ET.Element("Entry")
            instance = ET.Element("Instance")
            instance.text = f"{a}"
            prob_table = ET.Element("ProbTable")
            prob_table.text = "1"
            entry.append(instance)
            entry.append(prob_table)
            parameter.append(entry)

        func.append(var)
        func.append(parent)
        func.append(parameter)
        

    def set_obs(self, elem):
        cond_prob = ET.SubElement(elem, "CondProb")
        var = ET.Element("Var")
        var.text = "obs_int_behavior"
        parent = ET.Element("Parent")
        parent.text = "action_int_request int_time1" + " ".join([f"risk_{_}1" for _ in range(len(self.risk_positions))])
        parameter = ET.Element("Parameter", attrib={"type":"TBL"})
        for a in self.action_int_request:
            for int_time in self.s_int_time:
                for risks in itertools.product([0, 1], repeat=len(self.risk_positions)):
                    entry = ET.Element("Entry")
                    instance = ET.Element("Instance")
                    instance.text = f"{a} {int_time}"  + " ".join([str(_) for _ in risks]) + " -"
                    acc = self.operator_model.int_acc(int_time)
                    prob_table = ET.Element("ProbTable")
                    if risks[a] == 1:
                        prob_table.text = f"{1.0 - acc} {acc}"
                    elif risks[a] == 0:
                        prob_table.text = f"{acc} {1.0 - acc}"
                    entry.append(instance)
                    entry.append(prob_table)
                    parameter.append(entry)


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
        prob_list[self.s_ego_pose.tolist().index(0.0)] = 1.0
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
        prob_list[self.s_ego_speed.tolist().index(max(self.s_ego_speed))] = 1.0
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
        prob_list[self.s_int_time.tolist().index(0)] = 1.0
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
        prob_list[self.s_int_target.tolist().index(-1)] = 1.0
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
            instance.text = "-"
            prob_table = ET.SubElement(entry, "ProbTable")
            risk = 0.5 
            prob_table.text = f"{risk} {1.0-risk}"


def trial_until_sat():

    with open(sys.argv[1]) as f:
        param = yaml.safe_load(f)
    
    filename = sys.argv[1].split(".")[-1]
    dp = MDP(param)
    tree = ET.ElementTree(element=dp.root)
    ET.indent(tree, space="    ", level=0)
    tree.write("out.pomdpx")
    
if __name__ == "__main__":
    trial_until_sat()
