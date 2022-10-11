#!/usr/bin/bash
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from dp_policy_agent import *
import random, copy

class QAgent(DpPolicyAgent):
    """次の行動が最適なものを選ぶという前提でQ関数を更新する（実際の行動は関係なし）なので、off-policyと呼ばれる。
    水たまりを避けるのは遅いが、sarsaに比べて水たまり以外の場所は最短経路を動くような方策を学習する
    """
    def __init__(self, time_interval, estimator, puddle_coef=100, alpha=0.5, width=np.array([0.2, 0.2, math.pi/18]).T, 
                 lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T, dev_borders=[0.1, 0.2, 0.4, 0.8]):
        super().__init__(time_interval, estimator, None, puddle_coef, width, lowerleft, upperright)

        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作っておく
        
        self.actions = list(set([tuple(self.policy_data[i]) for i in self.indexes])) # 全部のアクションから重複を抜く
        self.ss = self.set_action_value_function() # PuddleIgnorePolicyの読み込み
        
        self.alpha = alpha
        self.s, self.a = None, None # 11.1のs,a
        self.update_end = False

    def set_action_value_function(self):
        """状態価値観数を読み込んで行動価値関数を初期化
        """
        ss = {}
        for line in open("puddle_ignore_value.txt", "r"):
            d = line.split()
            index, value = (int(d[0]), int(d[1]), int(d[2])), float(d[3]) # indexをタプル、値を数値に
            ss[index] = StateInfo(len(self.actions)) # StateInfoオブジェクトを割り当てて初期化

            # 方策の行動価値を価値のファイルに書いてある値にする。方策(policy.txt)と一致しない行動の場合は少し減点
            for i, a in enumerate(self.actions):
                ss[index].q[i] = value if tuple(self.policy_data[index]) == a else value - 0.1

        return ss

    def policy(self, pose, goal=None):
        index = self.to_index(pose, self.pose_min, self.index_nums, self.width) # 姿勢をindexに変える
        s = tuple(index)
        a = self.ss[s].pi() # 行動価値関数を使って行動決定
        return s, a # 現状態のタプルと、行動のインデックス

    def decision(self, observation=None):
        if self.update_end:
            return 0.0, 0.0
        if self.in_goal:
            self.update_end = True

        # カルマンフィルタの実行
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(observation)

        # Puddleworldから受け取った情報から報酬を計算
        # KFの結果から、現在の状態と次の行動を取得
        s_, a_ = self.policy(self.estimator.pose)
        r = self.time_interval * self.reward_per_sec() # 状態遷移の報酬
        self.total_reward += r

        # Q学習と現在の状態、行動の保存
        self.q_update(self.s, self.a, r, s_)
        self.s, self.a = s_, a_

        # 出力
        self.prev_nu, self.prev_omega = self.actions[a_] # 自己位置推定用
        return self.actions[a_] # 速度、角速度のtuple

    def q_update(self, s, a, r, s_):
        """現状態の最大のQ値を持つ行動の価値(q_)を状態遷移後の価値と、実行した行動価値でQ値を更新する
        ただし、実際の行動は、epsilon-greedyで、ランダムな行動が交じるので、必ずしも更新に使った価値と、実際の行動の価値は一致しない
        sarsaは、実際に行った行動の価値を用いて更新している
        """
        if s == None:
            return 

        q = self.ss[s].q[a] # 更新前の価値
        q_ = self.final_value if self.in_goal else self.ss[s_].max_q() # ゴールにいるなら、状態価値は終端価値, 状態遷移後の価値
        self.ss[s].q[a] = (1.0 - self.alpha) * q + self.alpha * (r + q_) # 1.11 q:更新前の価値 q_:状態遷移後の価値 alpha:更新をどれだけ反映させるか

        with open("log.txt", "a") as f:
            f.write("{} {} {} prev_q:{:.2f}, next_step_max_q:{:.2f}, new_p:{:.2f}\n".format(s, r, s_, q, q_, self.ss[s].q[a]))
        

class SarsaAgent(DpPolicyAgent):
    """sarsaはepsilon_greddy(policy)で選択された行動の価値をQ関数の更新に使う。学習に使う方策自体を更新するので,on-policy. 行動の選択が価値の更新に影響を与えるので、experimentation-sensitive
    実際に行動した際の状態価値をQ値の更新に使うので、水たまりを避けるようになるのははやい（実際に悪い行動をして、悪い行動の原因となる状態には近づかないようになる）ただし、水たまり以外の場所では、変に蛇行するようになる。これは、epsilon_greddyによりふらつく→時間がかかるほどペナルティが上がる→水たまりでなくてもその場所のQ値が下がる
    """
    def __init__(self, time_interval, estimator, puddle_coef=100, alpha=0.5, width=np.array([0.2, 0.2, math.pi/18]).T, 
                 lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T, dev_borders=[0.1, 0.2, 0.4, 0.8]):
        super().__init__(time_interval, estimator, None, puddle_coef, width, lowerleft, upperright)

        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作っておく
        
        self.actions = list(set([tuple(self.policy_data[i]) for i in self.indexes])) # 全部のアクションから重複を抜く
        self.ss = self.set_action_value_function() # PuddleIgnorePolicyの読み込み
        
        self.alpha = alpha
        self.s, self.a = None, None # 11.1のs,a
        self.update_end = False

    def set_action_value_function(self):
        """状態価値観数を読み込んで行動価値関数を初期化
        """
        ss = {}
        for line in open("puddle_ignore_value.txt", "r"):
            d = line.split()
            index, value = (int(d[0]), int(d[1]), int(d[2])), float(d[3]) # indexをタプル、値を数値に
            ss[index] = StateInfo(len(self.actions)) # StateInfoオブジェクトを割り当てて初期化

            # 方策の行動価値を価値のファイルに書いてある値にする。方策(policy.txt)と一致しない行動の場合は少し減点
            for i, a in enumerate(self.actions):
                ss[index].q[i] = value if tuple(self.policy_data[index]) == a else value - 0.1

        return ss

    def policy(self, pose, goal=None):
        index = self.to_index(pose, self.pose_min, self.index_nums, self.width) # 姿勢をindexに変える
        s = tuple(index)
        a = self.ss[s].pi() # 行動価値関数を使って行動決定
        return s, a # 現状態のタプルと、行動のインデックス

    def decision(self, observation=None):
        if self.update_end:
            return 0.0, 0.0
        if self.in_goal:
            self.update_end = True

        # カルマンフィルタの実行
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(observation)

        # Puddleworldから受け取った情報から報酬を計算
        # KFの結果から、現在の状態と次の行動を取得
        s_, a_ = self.policy(self.estimator.pose)
        r = self.time_interval * self.reward_per_sec() # 状態遷移の報酬
        self.total_reward += r

        # Q学習と現在の状態、行動の保存
        self.q_update(self.s, self.a, r, s_)
        self.s, self.a = s_, a_

        # 出力
        self.prev_nu, self.prev_omega = self.actions[a_] # 自己位置推定用
        return self.actions[a_] # 速度、角速度のtuple

    def q_update(self, s, a, r, s_):
        """epsilon_greddyで選択された行動の価値(q_値)と、実行した行動価値でQ値を更新する
        """
        if s == None:
            return 

        q = self.ss[s].q[a] # 更新前の価値
        q_ = self.final_value if self.in_goal else self.ss[s_].q(a_) # ゴールにいるなら、状態価値は終端価値, 状態遷移後の価値
        self.ss[s].q[a] = (1.0 - self.alpha) * q + self.alpha * (r + q_) # 1.11 q:更新前の価値 q_:状態遷移後の価値 alpha:更新をどれだけ反映させるか

        with open("log.txt", "a") as f:
            f.write("{} {} {} prev_q:{:.2f}, next_step_max_q:{:.2f}, new_p:{:.2f}\n".format(s, r, s_, q, q_, self.ss[s].q[a]))

class WarpRobot(Robot):
    """ランダムな場所にロボットを置く
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_agent = copy.deepcopy(self.agent)

    def choose_pose(self):
        """ゴールについたエージェントをどこかに置き直すための位置
        """
        xy = random.random()*6-2
        t = random.random()*2*math.pi
        return np.array([3, xy, t]).T if random.random() > 0.5 else np.array([xy, 3, t]).T

    def reset(self):
        """エージェントを置き直した時に、変数を初期化する
        """
        tmp = self.agent.ss # stete space は学習結果なので、使い回し
        self.agent = copy.deepcopy(self.init_agent)
        self.agent.ss = tmp

        self.pose = self.choose_pose()
        self.agent.estimator.belief = multivariate_normal(mean=self.pose, cov=np.diag([1e-10, 1e-10, 1e-10]))

        self.poses = [] # 軌跡を消去

    def one_step(self, time_interval):
        if self.agent.update_end:
            with open("log.txt", "a") as f:
                f.write("{}\n".format(self.agent.total_reward + self.agent.final_value))
            self.reset()
            return

        super().one_step(time_interval)


class StateInfo:
    """あるセルにおけるQ関数を管理する
    Q関数：ある状態における全actionに対する行動価値
    """
    def __init__(self, action_num, epsilon=0.3):
        """
        action_num: 行動の種類
        """
        self.q = np.zeros(action_num)
        self.epsilon = epsilon

    def greedy(self):
        """最も価値の高い行動のインデックスを返す
        """
        return np.argmax(self.q)

    def epsilon_greedy(self, epsilon):
        """epsilonの確率でランダムに行動を選択する
        """
        if random.random() < epsilon:
            return random.choice(range(len(self.q)))
        else:
            return self.greedy()

    def pi(self):
        return self.epsilon_greedy(self.epsilon)

    def max_q(self):
        return max(self.q)

def trial():
    time_interval = 0.1
    world = PuddleWorld(400000, time_interval, debug=False)

    m = Map()
    for ln in [(-4, 2), (2, -3), (3, 3), (-4, -4)]:
        m.append_landmark(Landmark(*ln))
        world.append(m)

    g = Goal(-3,-3)
    world.append(g)
    world.append(Puddle((-2, 0), (0, 2), 0.1))
    world.append(Puddle((-0.5, -2), (2.5, 1), 0.1))
    
    init_pose = np.array([3, 3, 0]).T
    kf = KalmanFilter(m, init_pose)
    a = QAgent(time_interval, kf)
    r = WarpRobot(init_pose, sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0), agent=a, color="red", bias_rate_stds=(0, 0))
    world.append(r)

    world.draw()
    return a

a = trial()
