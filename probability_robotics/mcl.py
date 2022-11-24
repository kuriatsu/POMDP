#!/use/bin/python3
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from robot import *
from scipy.stats import multivariate_normal
import random
import copy

class Particle:
    """パーティクルを管理するclass
    """
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight # 観測後のパーティクルの尤度による重み付け

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        """状態遷移関数のパーティクル一つ一つの移動。
        ガウス分布に従って制御にノイズを乗せて移動させる。

        """
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose) # noisedの速度と角速度でパーティクルの状態を更新
        
    def observation_update(self, obseravtion, envmap, distance_dev_rate, direction_dev):
        for d in obseravtion:
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルの位置と地図からランドマークの距離と方角を計算
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map) # particleの位置とランドマークの観測結果から、particleから見たランドマークの位置（パーティクルによる観測）を計算

            # 尤度計算
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2])) # 5.23
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos) # 5.24 共分散covの２次元ガウス分布を求め、obs_posの観測確率(self.poseの尤度)を計算
class Mcl:
    """パーティクルを管理するmcl推定機, numとパーティクルの初期位置から、Particleのリストを作る
    """
    def __init__(self, envmap, init_pose, num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05, # 1000回の計測実験から得られた、センサの距離と確度に対する標準偏差→尤度を求める
                 ):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)] # パーティクルの初期位置

        self.map = envmap
        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2]) # 与えられたリストの要素を対角成分に持つ対角行列,直進により生じる道のりのばらつきの標準偏差、回転で生じる道のりのry), 直進により生じる向きのry), 回転により生じる向きのry)
        self.motion_noise_rate_pdf = multivariate_normal(cov=c) # σ_abの４変数に対応した４次元のガウス分布を作成
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        self.ml = self.particles[0]
        self.pose = self.ml.pose

    def set_ml(self):
        """observation_updateで呼び出される
        """
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose

    def motion_update(self, nu, omega, time):
        """パーティクルを動かす、状態遷移関数。
        パーティクルの集合が制御後の信念b^
        """
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf) # Particleのmotion_update

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles] # パーティクルの矢印描写のための、パーティクルの向きx成分
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles] # パーティクルの矢印描写のための、パーティクルの向きy成分, 重みもかける、パーティクルが増えると、１つあたりの重みの平均値が小さくなるので、パーティクルの数もかける
        elems.append(ax.quiver(xs, ys, vxs, vys, angles="xy", scale_units="xy", scale=1.5, color="blue", alpha=0.5)) # 矢印

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.systematic_resampling() # 重みに基づいて確率的にリサンプリング
        
    def resampling(self):
        """重みに基づいて確率的にリサンプリングする。
        最後に重みをすべて1.0/numにすることで、パーティクルの重みの偏りを防ぐ
        """
        ws = [e.weight for e in self.particles]
        if sum(ws) < 1e-100:
            ws = [e + 1e-100 for e in ws] # wsの和が０に丸め込まれるとエラーになるので、対策
        ps = random.choices(self.particles, weights=ws, k=len(self.particles)) # wsの要素に比例した確率で、num個のパーティクルを選ぶ
        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)

    def systematic_resampling(self):
        """系統サンプリング
        計算量がO(N)でサンプリングバイアスが無い（同じ重みのパーティクルが均等に選択される。無駄にパーティクルが消えない）
        1. 重みを積み上げたようなリストを作成
        2. r = U(0, W/N) W:重み合計、1の末端
        3. 1の頭からrの位置にポインタを置き、その要素のもととなるパーティクルを選択
        4. r += W/N(<-step)
        """
        ws = np.cumsum([e.weight for e in self.particles]) # 重みを累積して足していく
        if ws[-1] < 1e-100:
            ws = [e + 1e-100 for e in ws]
        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)
            
class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator  = estimator
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        """Mcl(estimator)のmotion_updateを実行する(状態遷移関数により操作後の信念b^)
        observation_update->resamplingで観測により信念を更新
        Robotのone_stepで実行される
        """
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval) # 1つ前の制御指令値でパーティクルの姿勢を更新する
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.prev_nu, self.prev_omega

    def draw(self, ax, elems):
        """Robotのagentとして、インスタンス化、Robotのdraw()で呼ばれる。
        """
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t*180/math.pi)%360) 
        elems.append(ax.text(x, y+0.1, s, fontsize=10))

world = World(30, 0.1)

m = Map()
for ln in [(-4, 2), (2, 3), (3, 3)]:
    m.append_landmark(Landmark(*ln))

# world.append(m)
# initial_pose = np.array([2, 2, math.pi/6]).T
# estimator = Mcl(initial_pose, 100)
# circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)
# r = Robot(initial_pose, sensor=Camera(m), agent=circling)

# world.append(r)

# world.draw()

# # test motion_noise_std
# initial_pose = np.array([0, 0, 0]).T
# estimator = Mcl(initial_pose, 100, motion_noise_stds={"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})
# a = EstimationAgent(0.1, 0.2, 10.0/180*math.pi, estimator)
# estimator.motion_update(0.2, 10.0/180*math.pi, 0.1)
# for p in estimator.particles:
#     print(p.pose)

# def trial_particle_filter(motion_noise_stds):
def trial_particle_filter():
    time_interval = 0.1
    world = World(30, time_interval)

    initial_pose = np.array([0, 0, 0]).T
    estimator = Mcl(initial_pose, 100, motion_noise_stds)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=None, agent=circling, color="red")
    world.append(r)
    world.draw()

def trial():
    time_interval = 0.1
    world = World(30, time_interval)
    m = Map()
    for ln in [(-4, 2), (2, -3), (3, 3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([0, 0, 0]).T
    estimator = Mcl(m, initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    world.draw()

if __name__=="__main__":
    # trial({"nn":0.19, "no":0.001, "on":0.13, "oo":0.2})
    trial()
