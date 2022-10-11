#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from puddle_world import *
import itertools
import seaborn as sns
import collections 

class PolicyEvaluator:
    def __init__(self, width, goal, puddles, time_interval, sampling_num, puddle_coef=100.0, lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T):
        self.pose_min = np.r_[lowerleft, 0] # poseの最小値は-4, -4, 0[rad]
        self.pose_max = np.r_[upperright, math.pi*2] # poseの最大値は4, 4, 2π thetaの範囲0-2πで固定
        self.width = width
        self.goal = goal

        self.index_nums = ((self.pose_max - self.pose_min)/self.width).astype(int)
        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作っておく

        self.time_interval = time_interval
        self.puddle_coef = puddle_coef

        self.value_function, self.final_state_flags = self.init_value_function()
        self.policy = self.init_policy()
        self.actions = list(set([tuple(self.policy[i]) for i in self.indexes]))# policyの行動をsetいすることで重複を削除して、リストにする
        self.state_transition_probs = self.init_state_transition_probs(time_interval, sampling_num)
        self.depths = self.depth_means(puddles, sampling_num)

    def policy_evaluation_sweep(self):
        """終端状態以外の価値関数を、行動価値（行動後のpuddlesと時間ペナルティに遷移確率をかけたもの）に基づいて計算
        全離散状態（終端状態以外）に対して、action_valueを実行して状態価値を計算することを、sweepと呼ぶ
        """
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                q = self.action_value(tuple(self.policy[index]), index)
                delta = abs(self.value_function[index] - q)
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = q
                
        return max_delta

    def action_value(self, action, index):
        """一つのセルについて、行動価値を計算(10.17)
        """
        value = 0.0
        for delta, prob in self.state_transition_probs[(action, index[2])]: # 方向のインデックスindex[2]
            after = tuple(self.out_correction(np.array(index).T + delta))
            reward = -self.time_interval * self.depths[(after[0], after[1])] * self.puddle_coef - self.time_interval
            value += (self.value_function[after] + reward) * prob # 10.17

        return value

    def out_correction(self, index):
        """方角の処理
        indexが状態遷移によって範囲をはみ出たときに処理する。θを正規化してindexを返す、x,yのindexの処理は、今回の方策でははみ出すことがないので未処理
        """
        index[2] = (index[2] + self.index_nums[2])%self.index_nums[2]

        return index

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

puddles = [Puddle((-2, 0), (0,2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)] 
pe = PolicyEvaluator(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, 3), puddles, 0.1, 10)
# sns.heatmap(np.rot90(v), square=False) # ｘ軸が行、ｙが列になっているので、世界座標系に変換
# sns.heatmap(np.rot90(pe.depths), square=False) # puddleの離散化の結果を可視化
delta = 1e100
counter = 0
while delta > 0.01:
    delta = pe.policy_evaluation_sweep()
    counter += 1
    print(counter, delta)

v = pe.value_function[:, :, 18] # 向きtheta=18の時の状態価値関数
sns.heatmap(np.rot90(v), square=False) # puddleの離散化の結果を可視化
plt.show()
