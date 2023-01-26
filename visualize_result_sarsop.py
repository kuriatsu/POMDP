#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = [8, 12]
risk_colors = ["red", "blue", "green"]
trajectory_color = "green"
title_list = [r"$\theta_{\mathrm{int}}=\mathrm{Low}$", r"$\theta_{\mathrm{int}}=\mathrm{Mid}$", r"$\theta_{\mathrm{int}}=\mathrm{High}$"]

df_low = pd.read_csv(sys.argv[1]).dropna()
df_mid = pd.read_csv(sys.argv[2]).dropna()
df_high = pd.read_csv(sys.argv[3]).dropna()

sns.set(context='paper', style='whitegrid')
fig, ax = plt.subplots(6, 1)

for r, df in enumerate([df_low, df_mid, df_high]):
    ax_driving = ax[2*r]
    ax_belief = ax[2*r+1]
    ax_belief.set_ylim((0.0, 1.0))

    for i, p in enumerate([80, 120]):
        ax_driving.axvspan(p-1, p+1, color=risk_colors[i], alpha=1.0, label="target "+str(i+1))
        ax_belief.plot(df.mileage, df["risk_"+str(i+1)], c=risk_colors[i], label="belief "+str(i+1))
   
    for i, row in df.iterrows():

        if row.action_request != -1:
            ax_belief.axvspan(row.mileage, df.iloc[i+1].mileage, color=risk_colors[int(row.action_request)], alpha=0.2, label="intervention request")


    ax_driving.plot(df.mileage, df.speed, c=trajectory_color, label="speed")
    ax_driving.set_ylabel("speed [m/s]", fontsize=14)
    ax_driving.set_title(title_list[r], fontsize=14)
    ax_belief.set_ylabel("belief \nprobability", fontsize=14)
    ax_belief.set_xlabel("mileage [m]", fontsize=14)
ax[-1].legend() 
ax[-2].legend() 
plt.show()
