#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from puddle_world import *
import itertools
import seaborn as sns
import collections 

class DynamicProgramming:
    def __init__(self, width, goal, puddles, time_interval, sampling_num, puddle_coef=100.0, lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T):
        self.pose_min = np.r_[lowerleft, 0] # poseの最小値は-4, -4, 0[rad]
        self.pose_max = np.r_[upperright, math.pi*2] # poseの最大値は4, 4, 2π thetaの範囲0-2πで固定
        self.width = width
        self.goal = goal

        self.index_nums = ((self.pose_max - self.pose_min)/self.width).astype(int) # セルのインデックス
        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作っておく

        self.time_interval = time_interval
        self.puddle_coef = puddle_coef

        self.value_function, self.final_state_flags = self.init_value_function()
        self.policy = self.init_policy() # 初期値は、puddleを無視してゴールに向かうpolicy
        self.actions = list(set([tuple(self.policy[i]) for i in self.indexes]))# 全セルにおける初期policyによるactionを洗い出し、setすることで重複を削除して、全actionリストを作成
        self.state_transition_probs = self.init_state_transition_probs(time_interval, sampling_num)
        self.depths = self.depth_means(puddles, sampling_num)

    def value_iteration_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                max_q = -1e100
                max_a = None
                qs = [self.action_value(a, index) for a in self.actions] # 全行動の行動価値を計算
                max_q = max(qs) # 最大の行動価値
                max_a = self.actions[np.argmax(qs)] # 最大の行動価値を与える行動

                delta = abs(self.value_function[index] - max_q)
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = max_q # 価値の更新
                self.policy[index] = np.array(max_a).T # 方策の更新

        return max_delta

    def policy_evaluation_sweep(self):
        """終端状態以外の価値関数を、行動価値（行動後のpuddlesと時間ペナルティに遷移確率をかけたもの）に基づいて計算
        全離散状態（終端状態以外）に対して、action_valueを実行して状態価値を計算することを、sweepと呼ぶ
        """
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                q = self.action_value(tuple(self.policy[index]), index, out_penalty=False) # indexのセルにおけるactionの行動価値
                delta = abs(self.value_function[index] - q) # 存の価値関数との変化量を算出
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = q
                
        return max_delta

    def action_value(self, action, index, out_penalty=True):
        """一つのセルについて、行動価値を計算(10.17)
        """
        value = 0.0
        for delta, prob in self.state_transition_probs[(action, index[2])]: # 方向のインデックスindex[2]
            after, out_reward = self.out_correction(np.array(index).T + delta) # actionによる遷移後の状態と、外に出た場合のpenalty
            after = tuple(after)
            reward = -self.time_interval * self.depths[(after[0], after[1])] * self.puddle_coef - self.time_interval + out_reward*out_penalty
            value += (self.value_function[after] + reward) * prob # 10.17 ベルマン方程式

        return value

    def out_correction(self, index):
        """方角の処理
        indexが状態遷移によって範囲をはみ出たときに処理する。θを正規化してindexを返す、x,yのindexの処理は、puddleIgnoreAgentの方策でははみ出すことがないので未処理だが、行動価値観数による方策でははみ出す可能性があるので、範囲外に出たときに大きなペナルティを与え、範囲内に状態遷移をもとに戻す処理も加える。
        """
        out_reward = 0.0
        index[2] = (index[2] + self.index_nums[2])%self.index_nums[2] # 回転のインデックスがはみ出さないように 

        for i in range(2):
            if index[i] < 0:
                index[i] = 0
                out_reward = -1e100
            elif index[i] >= self.index_nums[i]:
                index[i] = self.index_nums[i]-1
                out_reward = -1e100

        return index, out_reward

    def depth_means(self, puddles, sampling_num):
        """セルの中の座標を均等にsampling_num**2点サンプリングして、池の深さをリスト化
        離散化の影響で、puddlesの輪郭がぼやける
        """
        dx = np.linspace(0, self.width[0], sampling_num)
        dy = np.linspace(0, self.width[1], sampling_num)
        samples  = list(itertools.product(dx, dy))

        tmp = np.zeros(self.index_nums[0:2]) # 深さの合計が格納される
        for xy in itertools.product(range(self.index_nums[0]), range(self.index_nums[1])):
            for s in samples:
                pose = self.pose_min + self.width*np.array([xy[0], xy[1], 0]).T + np.array([s[0], s[1], 0]).T
                for p in puddles:
                    tmp[xy] += p.depth * p.inside(pose)

            tmp[xy] /= sampling_num**2 # 深さの合計からセル内の平均値に変換

        return tmp

    def init_state_transition_probs(self, time_interval, sampling_num):
        """セルの中の座標を均等にsampling_num**3点サンプリングする
        １つのあるセルから、どの角度でどの行動（３種類）を取ると、どの方向のセルに移動するのかの確率（遷移確率）を計算する
        """
        dx = np.linspace(0.001, self.width[0]*0.999, sampling_num) # セルを細切れにする。隣のセルにはみ出さないように端を避ける
        dy = np.linspace(0.001, self.width[1]*0.999, sampling_num)
        dt = np.linspace(0.001, self.width[2]*0.999, sampling_num)
        samples = list(itertools.product(dx, dy, dt))

        tmp = {}
        # 状態遷移確率は、動きは相対的な向きthetaとactionのみに依存するので、それらに対して相対的な遷移確率を求める
        # 各方向、行動でサンプリングした点を移動して(セルの）インデックスの増分を記録
        for a in self.actions:
            for i_t in range(self.index_nums[2]):
                transitions = []
                for s in samples:
                    before = np.array([s[0], s[1], s[2] + i_t*self.width[2]]).T + self.pose_min # 遷移前の姿勢
                    before_index = np.array([0, 0, i_t]).T # 遷移前のインデックス

                    after = IdealRobot.state_transition(a[0], a[1], time_interval, before)
                    after_index = np.floor((after - self.pose_min)/self.width).astype(int)

                    transitions.append(after_index - before_index)

                unique, count = np.unique(transitions, axis=0, return_counts=True) # 集計（どのセルへの移動が何回か）
                probs = [c/sampling_num**3 for c in count] #  サンプルで割って確率にする
                tmp[a, i_t] = list(zip(unique, probs))

        return tmp

    def init_policy(self):
        """各行動に対する制御指令値を格納sる配列。各離散状態の中心座標でのPuddleIgnoreAgentのpolicyメソッドの出力を格納（３種類の行動しか無い
        """
        tmp = np.zeros(np.r_[self.index_nums, 2])
        for index in self.indexes:
            center = self.pose_min + self.width*(np.array(index).T + 0.5)
            tmp[index] = PuddleIgnoreAgent.policy(center, self.goal)

        return tmp

    def init_value_function(self):
        """終端状態は終端価値、それ以外は適当な値を代入して、価値観数を計算
        """
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)

        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T) # 終端状態かを保存する配列
            v[index] = self.goal.value if f[index] else -100.0 # 終端状態の価値はgoal.value, それ以外は適当な値

        return v, f

    def final_state(self, index):
        """セルの四隅がgoalの中に入っていたら、そのセルが終端状態
        """
        x_min, y_min, _ = self.pose_min + self.width*index # 左下の座標
        x_max, y_max, _ = self.pose_min + self.width*(index+1) # 右上の座標

        corners = [[x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _]]
        return all([self.goal.inside(np.array(c).T) for c in corners]) # 全部のgoal.insideがTrue

def trial_until_sat():
    puddles = [Puddle((-2, 0), (0,2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)] 
    dp = DynamicProgramming(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, -3), puddles, 0.1, 10)
    # sns.heatmap(np.rot90(v), square=False) # ｘ軸が行、ｙが列になっているので、世界座標系に変換
    # sns.heatmap(np.rot90(pe.depths), square=False) # puddleの離散化の結果を可視化
    delta = 1e100
    counter = 0
    while delta > 0.01:
        delta = dp.value_iteration_sweep()
        counter += 1
        print(counter, delta)

    with open("policy.txt", "w") as f:
        for index in dp.indexes:
            p = dp.policy[index]
            f.write("{} {} {} {} {}\n".format(index[0], index[1], index[2], p[0], p[1]))

    with open("value.txt", "w") as f:
        for index in dp.indexes:
            p = dp.value_function[index]
            f.write("{} {} {} {} \n".format(index[0], index[1], index[2], p))

    v = dp.value_function[:, :, 18] # 向きtheta=18の時の状態価値関数
    sns.heatmap(np.rot90(v), square=False) # puddleの離散化の結果を可視化
    plt.show()

    p = np.zeros(dp.index_nums)
    for i in dp.indexes:
        p[i] = sum(dp.policy[i])

    sns.heatmap(np.rot90(p[:, :, 18]), square=False)
    plt.show()

def trial_one_time():
    
    puddles = [Puddle((-2, 0), (0,2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)] 
    dp = DynamicProgramming(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, -3), puddles, 0.1, 10)
    delta = 1e100
    counter = 0
    delta = dp.policy_evaluation_sweep()

    with open("puddle_ignore_policy.txt", "w") as f:
        for index in dp.indexes:
            p = dp.policy[index]
            f.write("{} {} {} {} {}\n".format(index[0], index[1], index[2], p[0], p[1]))

    with open("puddle_ignore_value.txt", "w") as f:
        for index in dp.indexes:
            p = dp.value_function[index]
            f.write("{} {} {} {} \n".format(index[0], index[1], index[2], p))

if __name__=="__main__":
    trial_until_sat()
