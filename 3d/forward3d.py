# NAME: 3D-acoustic wave equation forword modeling using 2-4 FD
# Author: Wang Juntao
# DATE: 2023.9.9 16:09

import time
import numpy as np
from utils import *
# FD
class Acoustic_FD:
    def __init__(self,time_model:time_model,spatial_model:spatial_model,source:source,receiver:receiver):
        self.time_model = time_model
        self.spatial_model = spatial_model
        self.source = source
        self.receiver = receiver
        self.a = [-2.5,4/3,-5/60]
        self.seis = np.zeros((self.time_model.nt,self.receiver.loc.shape[0]))
    def forward(self):
        # 中间参数
        alpha = (self.spatial_model.V * self.time_model.dt/self.spatial_model.spacing)**2
        beta = alpha*self.spatial_model.spacing**2
        P_t = np.zeros_like(self.spatial_model.V)
        P_t_1 = np.zeros_like(self.spatial_model.V)
        
        for i in range(self.time_model.nt):
            P = alpha * (
                self.a[1]*(np.roll(P_t,1,axis=0)+np.roll(P_t,1,axis=1)+np.roll(P_t,1,axis=2)+
                        np.roll(P_t,-1,axis=0)+np.roll(P_t,-1,axis=1)+np.roll(P_t,-1,axis=2))+
                self.a[2]*(np.roll(P_t,2,axis=0)+np.roll(P_t,2,axis=1)+np.roll(P_t,2,axis=2)+
                        np.roll(P_t,-2,axis=0)+np.roll(P_t,-2,axis=1)+np.roll(P_t,-2,axis=2))
            ) + (alpha*3*self.a[0]+2)*P_t - P_t_1
            P[self.source.loc] -= beta[self.source.loc] * self.source.wave.ricker[i]
            # save results
            for ir in range(self.receiver.loc.shape[0]):
                self.seis[i,ir] = P[tuple(self.receiver.loc[ir])]
            P_t_1 = P_t
            P_t = P
        return self.seis
