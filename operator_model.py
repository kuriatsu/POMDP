#!/usr/bin/python3
# -*-coding:utf-8-*-
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class OperatorModel:
    def __init__(self, min_time, min_time_var, acc_time_min, acc_time_var, acc_time_slope)
        self.min_time = min_time
        self.min_time_var = min_time_var
        self.acc_time_min = acc_time_min
        self.acc_time_var = acc_time_var
        self.acc_time_slope = acc_time_slope
        self.acc_list = [0.5, 0.75, 1.0]

    def int_acc_prob(self, int_time):
        """return intervention acc list[[int_acc, prob],...]
        """
        int_acc_mean = min(1.0, max(0.0, self.acc_time_min + self.acc_time_slope*(self.int_time-self.min_time))) 
        int_acc_norm = stats.norm(loc=int_acc_mean, scale=self.acc_time_var)
        min_time_norm = stats.norm()
        acc_prob = [] 
        acc_for_cdf = np.linspace(acc_list[0], acc_list[-1], len(acc_list)+1)

        # mid probability is whithin the area
        for i in range(1, len(self.acc_list)-1):
            print(acc_for_cdf[i])
            prob = int_acc_norm.cdf(acc_for_cdf[i+1]) - int_acc_norm.cdf(acc_for_cdf[i]
            acc_prob.append([self.acc_list[i], prob])
        
        # side probability is cumulative from -inf/inf 
        acc_prob.append([self.acc_list[0], int_acc_norm.cdf(acc_for_cdf[1])])
        acc_prob.append([self.acc_list[-1], 1.0 - int_acc_norm.cdf(acc_for_cdf[-2])])

        # no interventioon
        min_time_norm = stats.norm(loc=self.min_time, scale=self.min_time_var)
        acc_prob.append([None, 1.0 - min_time_norm.cdf(self.int_time+0.5)]) 

        return acc_prob

    def int_acc(self, int_time):
        intercept_time = 3
        intercept_acc = 0.5
        slope = 0.25

        if int_time < self.min_time:
            acc = None
        else:
            acc = self.min_time + self.acc_time_slope*(int_time - self.min_time)
            acc = min(acc, 1.0)
        
        return [[acc, 1.0]]


if __name__=="__main__":
    min_time = 3
    min_time_var = 0.5
    acc_time_min = 0.5
    acc_time_var = 0.2
    acc_time_slope = 0.2
    operator_model = OperatorModel(
        min_time,
        min_time_var,
        acc_time_min,
        acc_time_var,
        acc_time_slope
        )
    acc = operator_model.int_acc_prob(4)
   
    print(acc)
    data_acc = []
    for x in np.linspace(min_time, 6, 20):
        int_acc_mean = min(1.0, max(0.0, acc_time_min + acc_time_slope*(x-min_time))) 
        y_list = stats.norm.rvs(loc=int_acc_mean, scale=acc_time_var, size=100)
        for y in y_list:
            data_acc.append([x, y])

    data_min_time = []
    min_time_list = stats.norm.rvs(loc=min_time, scale=min_time_var, size=100)
    for y in np.linspace(0, acc_time_min, 20):
        for x in min_time_list:
            data_acc.append([x, y])

    data = np.array(data_acc)
    sns.scatterplot(x=data[:, 0], y=data[:, 1])
    plt.xlim(0.0, 6.0)
    plt.ylim(0.0, 1.0)
    plt.show()
