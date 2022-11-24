#!/usr/bin/python3
# -*-coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x_list = np.arange(0.0, 6.1, 0.1)
y_list_list = []
min_int_time_list = np.arange(0, 8, 2)
min_acc_list = np.arange(0.0, 1.25, 0.25) 
acc_slope_list = np.arange(0.0, 1.25, 0.25) 

for min_int_time in min_int_time_list:
    for min_acc in min_acc_list:
        for acc_slope in acc_slope_list:
            y_list = []
            for x in x_list:
                y_list.append(min(max(acc_slope * (x - min_int_time) + min_acc, 0.0), 1.0))
                 
            y_list_list.append(np.array(y_list))


fig, ax = plt.subplots()

for y_list in y_list_list:
    sns.lineplot(x=x_list, y=y_list, ax=ax)
plt.show()
