#!/usr/bin/python3
# -*-coding:utf-8-*-
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

class OperatorModel:
    def __init__(self, min_time, min_time_var, acc_time_min, acc_time_var, acc_time_slope, int_time_list):
        self.min_time = min_time
        self.min_time_var = min_time_var
        self.acc_time_min = acc_time_min
        self.acc_time_var = acc_time_var
        self.acc_time_slope = acc_time_slope
        self.acc_list = [0.5, 0.75, 1.0]
        
        self.int_time_list = int_time_list
        self.inttime_acc_prob_list = {}
        
        self.init_performance_model()


    def init_performance_model(self):
        self.inttime_acc_prob_list = {}
        for int_time in self.int_time_list:
            self.inttime_acc_prob_list[int_time] = self.calc_acc_prob(int_time)
            
    def calc_acc_prob(self, int_time):
        """return intervention acc list[[int_acc, prob],...]
        """

        # no interventioon
        min_time_norm = stats.norm(loc=self.min_time, scale=self.min_time_var)
        intervention_prob = min_time_norm.cdf(int_time) # NOTE: 0.5 was added to int_time

        int_acc_mean = min(1.0, max(0.0, self.acc_time_min + self.acc_time_slope*(int_time-self.min_time))) 
        int_acc_norm = stats.norm(loc=int_acc_mean, scale=self.acc_time_var)
        acc_prob = [] # [[acc, prob]]
        acc_for_cdf = np.linspace(self.acc_list[0], self.acc_list[-1], len(self.acc_list)+1)

        # mid probability is whithin the area
        for i in range(1, len(self.acc_list)-1):
            prob = (int_acc_norm.cdf(acc_for_cdf[i+1]) - int_acc_norm.cdf(acc_for_cdf[i]))*intervention_prob
            acc_prob.append([self.acc_list[i], prob])
        # side probability is cumulative from -inf/inf 
        acc_prob.append([self.acc_list[0], int_acc_norm.cdf(acc_for_cdf[1])*intervention_prob])
        acc_prob.append([self.acc_list[-1], (1.0 - int_acc_norm.cdf(acc_for_cdf[-2]))*intervention_prob])
        acc_prob.append([None, 1.0 - intervention_prob]) 


        return acc_prob

    def get_acc_prob(self, int_time):
        return self.inttime_acc_prob_list[int_time]

    def int_acc(self, int_time):

        if int_time < self.min_time:
            acc = None
        else:
            acc = self.min_time + self.acc_time_slope*(int_time - self.min_time)
            acc = min(acc, 1.0)
        
        return [[acc, 1.0]]

    
if __name__=="__main__":

    # min_time = 3
    # min_time_var = 0.5
    # acc_time_min = 0.5
    # acc_time_var = 0.1
    # acc_time_slope = 0.2
    int_time_list = [0, 1, 2, 3, 4, 5, 6]
    param_list = [
            "param_2_1.yaml",
            # "param_3_1.yaml",
            # "param_4_1.yaml",
            "param_6_1.yaml",
            "param_7_1.yaml",
            ]
    fig, axes = plt.subplots(1, len(param_list))
    for i, param in enumerate(param_list):

        with open(param) as f:
            param = yaml.safe_load(f)

        operator_model = OperatorModel(
            param["min_time"],
            param["min_time_var"],
            param["acc_time_min"],
            param["acc_time_var"],
            param["acc_time_slope"],
            int_time_list,
            )
        acc = operator_model.get_acc_prob(4)
        print(acc)
        operator_model.acc_list = np.arange(0.0, 1.25, 0.25).tolist()
        operator_model.init_performance_model()
        
        acc_mat = np.zeros((len(int_time_list), len(operator_model.acc_list)))
        for int_time in int_time_list:
            acc_prob_list = operator_model.get_acc_prob(int_time)
            for acc_prob in acc_prob_list:
                # print(acc_prob)
                if acc_prob[0] is None:
                    acc_mat[int_time,0] = acc_prob[1]
                else:
                    acc_mat[int_time, int((acc_prob[0]/0.25))] = acc_prob[1]
        sns.heatmap(np.rot90(acc_mat), ax=axes[i])
        axes[i].set_yticklabels([round(i, 2) for i in np.arange(1.0, -0.25, -0.25).tolist()])
        axes[i].set_xlabel("intervention time [s]", fontsize=14)
        axes[i].set_ylabel("intervention accuracy", fontsize=14)
    plt.show()
