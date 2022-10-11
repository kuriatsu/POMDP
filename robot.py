#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from ideal_robot import *
from scipy.stats import expon, norm, uniform

class Robot(IdealRobot):

    def __init__(self, pose, agent=None, sensor=None, color="black", 
                 noise_per_meter=5, noise_std=math.pi/60,
                 bias_rate_stds=(0.1, 0.1), 
                 expected_stuck_time=1e100, expected_escape_time=1e-100,
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0, 5.0), kidnap_range_y=(-5.0, 5.0)
                 ):
        """
        小石は1mに5個, 標準偏差は3deg, ロボットの向きthetaに影響。
        速度、角速度に対するバイアスを標準偏差10%で選択
        """
        super().__init__(pose, agent, sensor, color)
        # noise
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter)) # scele:小石を踏むまでの平均の道のり, 確率的に決定しており、指数分布に従うp(x|λ)=λ*e^(-λ*x),λは道のりあたりに小石を踏む期待値 。1e-100はnoise_per_meterに0が指定された時の対策。
        self.distance_until_noise = self.noise_pdf.rvs() # 小石を踏むまでの距離を確率分布から抽出
        self.theta_noise = norm(scale=noise_std) # thetaに加えるためのノイズの分布（ガウス分布）
        # bias
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0]) # バイアス割合決定
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1]) # バイアス割合決定
        # stuck 
        self.stuck_pdf = expon(scale=expected_stuck_time) # スタックまでの時間期待値の確率密度関数
        self.escape_pdf = expon(scale=expected_escape_time) # スタック脱出までの時間期待値の確率密度関数
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False
        # kidnap
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi))
        

    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu) * time_interval + self.r * abs(omega) * time_interval # 進んだ分だけ直進方向、回転方向の走行距離を引く
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs() # 小石を踏んだら、セットし直し
            pose[2] += self.theta_noise.rvs() # thetaにノイズを加える

        return pose

    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega

    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck) # スタックしてたら速度、角速度は0

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose

    def one_step(self, time_interval):
        """IdealRobotのものをコピー。ノイズ追加だけ
        """
        if not self.agent:
            return

        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega) # バイアス追加:
        nu, omega = self.stuck(nu, omega, time_interval) # スタック追加
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval) # ここだけ追加
        self.pose = self.kidnap(self.pose, time_interval)


class Camera(IdealCamera):
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range =(-math.pi/3, math.pi/3),
                 distance_noise_rate=0.1, direction_noise=math.pi/90,
                 distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi/90,
                 phantom_prob=0.0, phantom_range_x=(-5.0, 5.0), phantom_range_y=(-5.0, 5.0),
                 oversight_prob = 0.1,
                 occulusion_prob = 0.0,
                 ):
        super().__init__(env_map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev) # センサ距離バイアス
        self.direction_bias = norm.rvs(scale=direction_bias_stddev) # センサ確度バイアス
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1]-rx[0], ry[1]-ry[0]))
        self.phantom_prob = phantom_prob # 各ランドマークのファントムが出現する確率
        self.oversight_prob = oversight_prob
        self.occulusion_prob = occulusion_prob

    def noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T

    def bias(self, relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std, self.direction_bias]).T

    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
    
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos

    def occulusion(self, relpos):
        if uniform.rvs() < self.occulusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[0] - relpos[0]) # 現在のランドマーク距離とセンサの認識範囲6mの範囲内で、センサ値が実際よりも大きくなる
            phi = relpos[1]
            return np.array([ell, relpos[1]]).T
        else:
            return relpos

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            p = self.phantom(cam_pose, p)
            p = self.occulusion(p)
            p = self.oversight(p)
            if self.visible(p):
                p = self.bias(p)
                p = self.noise(p)
                observed.append((p, lm.id))

        self.lastdata = observed
        return observed

if __name__ == "__main__":
    world = World(20, 0.1)

    #for i in range(100):
    #    circling = Agent(0.2, 10.0/180*math.pi)
    #    r = Robot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")
    #    world.append(r)

    circling = Agent(0.2, 10.0/180*math.pi)
    gt_robot = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray")
    noise_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="orange", noise_per_meter=5, bias_rate_stds=[0.0, 0.0], expected_stuck_time=0.0, expected_escape_time=0.0)
    bias_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="green", noise_per_meter=0, bias_rate_stds=[0.2, 0.2], expected_stuck_time=0.0, expected_escape_time=0.0)
    noise_bias_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red", noise_per_meter=5, bias_rate_stds=[0.2, 0.2], expected_stuck_time=0.0, expected_escape_time=0.0)
    stuck_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="purple", noise_per_meter=0, bias_rate_stds=[0.0, 0.0], expected_stuck_time=10.0, expected_escape_time=10.0)
    kidnap_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="yellow", noise_per_meter=0, bias_rate_stds=[0.0, 0.0], expected_kidnap_time=5.0)

    # world.append(gt_robot)
    # world.append(noise_robot)
    # world.append(bias_robot)
    # world.append(noise_bias_robot)
    # world.append(stuck_robot)
    # world.append(kidnap_robot)
    m = Map()
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)
    r = Robot(np.array([0,0,0]).T, sensor=Camera(m, phantom_prob=0.5, occulusion_prob=0.5), agent=circling)
    world.append(r)

    world.draw()
