#!/usr/bin/python3
# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nobias_matrix = np.array([
    [0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0, 0, 0, 0, 0],
    ])
bias_matrix = np.array([
    [0, 0, 0, 0, 1.0],
    [0, 0.5, 0.5, 0.5, 0],
    [1.0, 0, 0, 0, 0],
    ])

fig, ax = plt.subplots(1,2)
sns.heatmap(nobias_matrix, ax=ax[0], vmin=0.0, vmax=1.0)
sns.heatmap(bias_matrix, ax=ax[1], vmin=0.0, vmax=1.0)
ax[0].set_xlabel(r"$risk_{target}$", fontsize=14)
ax[0].set_ylabel(r"$\theta_{bias}=P(risk_{target}=high|\mathcal{S})$", fontsize=14)
ax[1].set_xlabel(r"$risk_{target}$", fontsize=14)
ax[1].set_ylabel(r"$\theta_{bias}=P(risk_{target}=high|\mathcal{S})$", fontsize=14)
ax[0].set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
ax[0].set_yticklabels([1.0, 0.5, 0.0])
ax[1].set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
ax[1].set_yticklabels([1.0, 0.5, 0.0])
plt.show()

