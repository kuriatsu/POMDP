#! /user/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("sensor_data/sensor_data_600.txt", delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))
data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align="left")
# plt.show()

data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
d.lidar.mean().plot()
# plt.show()

each_hour = {i:d.lidar.get_group(i).value_counts().sort_index() for i in range(24)} # 時間ごとにデータの出現頻度を作成
freqs = pd.concat(each_hour, axis=1)
freqs = freqs.fillna(0)
probs = freqs/len(data) # 全体の中のセンサ値の出現頻度
# sns.heatmap(probs)
p_t = pd.DataFrame(probs.sum()) # 各時間毎の出現確率
p_z = pd.DataFrame(probs.transpose().sum()) # 各データ値毎の出現確率

cond_z_t = probs/p_t[0] # 時間ごとにP(t)で割るとP(x|t)
cond_t_z = probs.transpose()/p_z[0] # 行と列を入れ替えてP(z)で割るとP(t|x)

# ベイズ定理確認
print("P(z=630) = ", p_z[0][630])
print("P(t=13) = ", p_t[0][13])
print("Bayes P(z=630|t=13) = ", cond_t_z[630][13]*p_z[0][630]/p_t[0][13])
print("answer = ", cond_z_t[13][630])

def bays_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value]*current_estimation[i])

    return new_estimation/sum(new_estimation) # 正規化

# センサ値から、どの時間に計測されたかをベイズ定理で予測する
plt.clf()
estimation = bays_estimation(630, p_t[0])
# plt.plot(estimation)
# plt.show()

# 複数のセンサ値から、どの時間に計測されたかを予測する
values = [630, 632, 636]
estimation = p_t[0]
for v in values:
    estimation = bays_estimation(v, estimation)

# plt.plot(estimation)
# plt.show()
