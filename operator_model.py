#!/usr/bin/python3
# -*-coding:utf-8-*-
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def intervention_acc(int_time, min_time, min_time_var, acc_time_min, acc_time_var, acc_time_slope):
    int_acc_mean = min(1.0, max(0.0, acc_time_min + acc_time_slope*(int_time-min_time))) 
    int_acc_norm = stats.norm(loc=int_acc_mean, scale=acc_time_var)
    min_time_norm = stats.norm()
    acc_prob = [] 
    acc_list = [0.5, 0.75, 1.0]
    acc_for_cdf = np.linspace(acc_list[0], acc_list[-1], len(acc_list)+1)

    # mid probability is whithin the area
    for i in range(1, len(acc_list)-1):
        print(acc_for_cdf[i])
        prob = int_acc_norm.cdf(acc_for_cdf[i+1]) - int_acc_norm.cdf(acc_for_cdf[i]
        acc_prob.append([acc_list[i], prob])
    
    # side probability is cumulative from -inf/inf 
    acc_prob.append([acc_list[0], int_acc_norm.cdf(acc_for_cdf[1])])
    acc_prob.append([acc_list[-1], 1.0 - int_acc_norm.cdf(acc_for_cdf[-2])])

    # no interventioon
    min_time_norm = stats.norm(loc=min_time, scale=min_time_var)
    acc_prob.append([None, 1.0 - min_time_norm.cdf(int_time+0.5)]) 

    return acc_prob


if __name__=="__main__":
    min_time = 3
    min_time_var = 0.5
    acc_time_min = 0.5
    acc_time_var = 0.2
    acc_time_slope = 0.2
    acc = intervention_acc(
            4, 
            min_time,
            min_time_var,
            acc_time_min,
            acc_time_var,
            acc_time_slope
            )
   
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
