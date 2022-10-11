#!/usr/bin/python3 
# -*- coding:utf-8 -*-

import sys
sys.path.append(".")
from mcl import *
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

def sigma_elipse(p, cov, n):
    """誤差楕円を返す
    """
    eig_vals, eig_vec = np.linalg.eig(cov) # 共分散行列の固有値
    ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0])/math.pi*180 # 楕円の傾き
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]), angle=ang, fill=False, color="blue", alpha=0.5) 

def matM(nu, omega, time, stds):
    """P143 (6.5)
    入力νωによる実際の出力ν'ω'の分布をガウス分布で近似。ノイズは進行距離と角度に対して定義されているので、ν'ω'の共分散を計算するには、変換して分散の線型性に基づいて足し合わせる
    """
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time, stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])

def matA(nu, omega, time, theta):
    """P143 (6.11)
    テイラー展開を用いて、f(x_t-1, u'_t) = f(x_t-1, u_t) + A_t(u'_t - u_t)と線形近似を行う（f()は前状態から入力により次状態への状態遷移関数）これにより、制御u'が指令値uからずれた量に対して、本来はx'は対象にならない（角速度があるので、歪む）が、A_t....の項はu'の誤差に対して、x'が線形になることを示しており、対称になる→ガウス分布に従う
    """
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[(stw - st)/omega, -nu/(omega**2)*(stw-st) + nu/omega*time*ctw],
                     [(-ctw + ct)/omega, -nu/(omega**2)*(-ctw + ct) + nu/omega*time*stw],
                     [0,                 time]
                     ])

def matF(nu, omega, time, theta):
    """P143 (6.15)
    信念分布はf(x', u)をx'で積分したいが、fの中にx'が入っているので、積分が難しい。→テイラー展開による線形近似で外に出す
    f(x_t-1, u_t) = f(μ_t-1, u_t) + F(x_t-1, μ_t-1)
    Fはfをμ_t-1まわりで、x_t-1で偏微分したヤコビ行列（μは前状態x_t-1 = (x, y, θ).Tの平均)
    """
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu/omega * (math.cos(theta + omega*time) - math.cos(theta))
    F[1, 2] = nu/omega * (math.sin(theta + omega*time) - math.sin(theta))
    return F
                    
def matH(pose, landmark_pos):
    """6.25
    観測方程式がｘに依存しているので、テイラー展開の１次近似で外に出す。Hは観測方程式(3.17)をμ周り（移動後平均状態）でxで偏微分したもの
    """
    mx, my = landmark_pos
    mux, muy, mut = pose
    q = (mux - mx) ** 2 + (muy - my)**2
    return np.array([[(mux - mx)/np.sqrt(q), (muy - my)/np.sqrt(q), 0.0], [(my - muy)/q, (mux - mx)/q, -1.0]])

def matQ(distance_dev, direction_dev):
    """6.29
    観測の尤度を計算するためのガウス分布（パーティクルフィルタと同じ）
    """
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


class KalmanFilter:
    def __init__(self, envmap, init_pose, motion_noise_stds={"nn":0.19, "no":0.01, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05,
                 ):
        self.belief = multivariate_normal(mean=init_pose, cov=np.diag([0.1, 0.2, 0.01])) # 信念を表すガウス分布
        self.pose = self.belief.mean
        self.motion_noise_stds = motion_noise_stds
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

    def motion_update(self, nu, omega, time):
        """状態遷移関数でb^計算
        """
        if abs(omega) < 1e-5:
            omega = 1e-5

        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, self.belief.mean[2])
        F = matF(nu, omega, time, self.belief.mean[2])
        self.belief.cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T) # 移動後の信念の共分散 F(移動前の信念の共分散行列)F.T + AMA.T
        self.belief.mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean) # 移動後の信念の平均は、移動前からの状態遷移関数に従う（今回は雑音のない理想的な運動モデル）
        self.pose = self.belief.mean


    def observation_update(self, observation):
        """観測により信念更新
        """
        for d in observation:
            z = d[0]
            obs_id = d[1]

            H = matH(self.belief.mean, self.map.landmarks[obs_id].pos)
            estimated_z = IdealCamera.observation_function(self.belief.mean, self.map.landmarks[obs_id].pos)
            Q = matQ(estimated_z[0]*self.distance_dev_rate, self.direction_dev)
            K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T))) # 6.40
            self.belief.mean += K.dot(z - estimated_z) # 6.41
            self.belief.cov = (np.eye(3) - K.dot(H)).dot(self.belief.cov) # 6.42
            self.pose = self.belief.mean

    def draw(self, ax, elems):
        e = sigma_elipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3) # beliefは３次元なので、共分散行列はx, yに関わる左上2x2の部分だけ取り出す
        elems.append(ax.add_patch(e))

        # θ方向の誤差の3σの範囲を計算、θの推定値として、μ-3σ、μ+3σの向きに線分を引く
        x, y, c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2, 2])*3
        xs = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys = [y + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5)

def trial2():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4, 2), (2, -3), (3, 3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([0, 0, 0]).T
    # estimator = Mcl(m, initial_pose, 100, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2})
    # a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    kf = KalmanFilter(m, initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw()

# trial2()

