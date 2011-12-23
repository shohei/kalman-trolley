#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
from scipy import stats
import scipy
import scipy.linalg
from pylab import *

dt=0.001

mean_ak = 0.0 #a_kの平均値
sigma_a = 1.0 #a_kの標準偏差

mean_z = 0.0 #z_kの平均値
sigma_z = 1.0 #z_kの標準偏差

I2 = np.matrix(np.identity(2)) #単位行列を準備

#状態方程式　x_k = F x_k-1 + w_k
#ノイズ w_k = G a_k
F=mat([[1,dt],[0,1]]) 
G=mat([[dt**2/2.0],[dt]]) 
#観測方程式 y_k = H x_k
H=mat([1,0])
Q=mat((sigma_a**2)*G*G.T)
R=sigma_z**2

x_init = 0.0
v_init = 0.0
x_k_init = mat([[x_init],[v_init]])
P_k_init = mat([[0,0],[0,0]]) #誤差の共分散の初期値はゼロ行列を仮定

max_t = 1000 # 計算終了時間
t_init = 1# 計算開始時間
x_k_hat_aposteriori = x_k_init #初期化
P_k_aposteriori = P_k_init#初期化
result=[]
time_vec =[]
position_vec=[]
true_position=[]
time=0
x_k_true=x_k_init#初期化（真の状態値）
observed_position = []
for i in range(t_init,max_t):
    a_k = np.matrix(stats.norm.rvs(mean_ak,sigma_a,size=1)).transpose()
    v_k = np.matrix(stats.norm.rvs(mean_z,sigma_z,size=1)).transpose()
    #この場合のa_kは入力となる
    #a_kをステップごとに更新している
    z_k = H*x_k_hat_aposteriori + v_k
    observed_position.append(z_k.tolist()[0])
#Time Update "Predict"
    x_k_hat_apriori = F * x_k_hat_aposteriori + G * a_k
    P_k_apriori = F*P_k_aposteriori*F.T + Q
#Record the true position
    x_k_true = F*x_k_true+G*a_k
    true_position.append(x_k_true.tolist()[0])
#Measurement Update "Correct"
    S_k = (H*P_k_apriori)*H.T+R #observation innovation
    K_k = P_k_apriori*H.T*S_k.I # Compute optimized Karman gain
    x_k_hat_aposteriori = x_k_hat_apriori + K_k*(z_k - H* x_k_hat_apriori)
    P_k_aposteriori = (I2 - K_k*H)*P_k_apriori
#Writing result
    time += dt
    position = x_k_hat_aposteriori.tolist()[0][0]
    time_vec.append(time)
    position_vec.append(position)

result = zip(time_vec,position_vec)
print "result",result
print true_position

plot(time_vec,true_position)
plot(time_vec,position_vec)
#plot(time_vec,observed_position)
#xlim(xmax=10)
legend(("true","karman filter"))
show()

