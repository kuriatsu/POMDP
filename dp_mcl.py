#!/usr/bin/python3
# -*-coding:utf-8 -*-

import sys
sys.path.append(".")
from dp_policy_agent import *
from dynamic_programming import *

class QmdpAgent(DpPolicyAgent):
    def __init__(self, time_interval, estimator, goal, puddles, sampling_num=10, width=np.array([0.2, 0.2, math.pi/18]).T,
                 puddle_coef=100.0, lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T):
        super().__init__(time_interval, estimator, goal, puddle_coef, width, lowerleft, upperright)
        self.dp = DynamicProgramming(width, goal, puddles, time_interval, sampling_num)
        self.dp.value_function = self.init_value()
        self.evaluations = np.array([0.0, 0.0, 0.0]) # QMDP値を入れるため
        self.current_value = 0.0 # 現在の状態価値、描画用
        self.history = [(0, 0)]

    def init_value(self):
        tmp = np.zeros(self.dp.index_nums)
        for line in open("value.txt", "r"):
            d = line.split()
            tmp[int(d[0]),int(d[1]),int(d[2])] = float(d[3])

        return tmp

    def evaluation(self, action, indexes):
        """QMDP計算
        パーティクルの存在するセルのQ値とパーティクルの重みをかけることで、パーティクルのQ値の期待値を算出。
        今回は、パーティクルの重みは正規化されているので、単純に平均を取る
        action_value:状態遷移確率によって算出された次状態の価値関数＋行動価値*遷移確率（MDPのやり方で計算、Q学習で計算したpolicyで計算していない）
        """
        return sum([self.dp.action_value(action, i, out_penalty=False) for i in indexes])/len(indexes) # パーティクルの重みの正規化が前提の計算方法

    def policy(self, pose, goal=None):
        """indexesにパーティクルが属するstate spaceのindexを格納
        state spaceのQMDP値をcurrent_valueとして計算
        すべてのactionの、QMDP値のリストを返す
        """
        indexes = [self.to_index(p.pose, self.pose_min, self.index_nums, self.width) for p in self.estimator.particles]
        self.current_value = sum([self.dp.value_function[i] for i in indexes])/len(indexes) # 描画用 現在のパーティクルすべての状態価値の平均
        self.evaluations = [self.evaluation(a, indexes) for a in self.dp.actions]
        self.history.append(self.dp.actions[np.argmax(self.evaluations)]) # 全アクションのリストから、最もQMDP値が高いactionを選択

        # ローカルミニマムでロボットがスタックした場合の対処
        if self.history[-1][0] + self.history[-2][0] == self.history[-1][1] + self.history[-2][1] == 0.0:
            return (1.0, 0.0)

        return self.history[-1]

    def draw(self, ax, elems):
        super().draw(ax, elems)
        # 現在のパーティクルの状態価値の平均→信念状態の平均（パーティクルが正規化されているので、重み付けする必要がない）+ (左回転、直進、右回転）のQMDP値を表示
        elems.append(ax.text(-4.5, -4.6, "{:.3f} => [{:.3f}, {:.3f}, {:.3f}]".format(self.current_value, *self.evaluations), fontsize=8))

def trial(animation):
    time_interval=0.1
    world = PuddleWorld(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(1,4), (4,1), (-4,-4)]:
        m.append_landmark(Landmark(*ln))

    world.append(m)

    goal = Goal(-3, -3)
    puddles = [Puddle((-2, 0), (0, 2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)]
    world.append(goal)
    world.append(puddles[0])
    world.append(puddles[1])

    init_pose = np.array([2.5, 2.5, 0]).T
    pf = Mcl(m, init_pose, 100)
    a = QmdpAgent(time_interval, pf, goal, puddles)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    return a

trial(True)
