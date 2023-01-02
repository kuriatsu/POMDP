#!/usr/bin/python3
# -*-coding:utf-8-*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("w_perf3/summary_rate.csv")
fig, ax = plt.subplots()
sns.barplot(data=data, x="target-agent", y="degrade", hue="theta_accuracy", ax=ax, palette=sns.color_palette(["orangered"]*3), edgecolor="0.2", hue_order=["low", "mid", "high"], order=["egoistical-myopic", "egoistical-pomdp", "myopic-pomdp"])
sns.barplot(data=data, x="target-agent", y="same", hue="theta_accuracy", ax=ax, palette=sns.color_palette(["gray"]*3), edgecolor="0.2", hue_order=["low", "mid", "high"], order=["egoistical-myopic", "egoistical-pomdp", "myopic-pomdp"])
sns.barplot(data=data, x="target-agent", y="trav_imp", hue="theta_accuracy", ax=ax, palette=sns.color_palette(["limegreen"]*3), edgecolor="0.2", hue_order=["low", "mid", "high"], order=["egoistical-myopic", "egoistical-pomdp", "myopic-pomdp"])
sns.barplot(data=data, x="target-agent", y="req_imp", hue="theta_accuracy", ax=ax, palette=sns.color_palette(["yellow"]*3), edgecolor="0.2", hue_order=["low", "mid", "high"], order=["egoistical-myopic", "egoistical-pomdp", "myopic-pomdp"])
sns.barplot(data=data, x="target-agent", y="improve", hue="theta_accuracy", ax=ax, palette=sns.color_palette(["turquoise"]*3), edgecolor="0.2", hue_order=["low", "mid", "high"], order=["egoistical-myopic", "egoistical-pomdp", "myopic-pomdp"])
ax.set_xticks([-0.3, 0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3])
ax.set_xticklabels(["low", "mid", "high","low", "mid", "high", "low", "mid", "high"], fontsize=14)
ax.set_xlabel("Policy", fontsize=14)
ax.set_ylabel("improve rate compared to myopic agent", fontsize=14)
handles, labels = ax.get_legend_handles_labels()
print(handles)
ax.legend(handles[::3], ["degrade", "keep", "shorter request time", "shorter travel time", "improve"], bbox_to_anchor=(1.0, 1.0), loc='lower right', fontsize=14)

plt.show()

