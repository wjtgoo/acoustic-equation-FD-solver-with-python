# NAME: 3D-acoustic wave equation forword modeling using 2-4 FD by pytorch
# Author: Wang Juntao
# DATE: 2023.9.10 16:59

import time
import numpy as np
import torch
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# FD
class Acoustic_FD:
    def __init__(self,time_model:time_model,spatial_model:spatial_model,source:source,receiver:receiver):
        self.time_model = time_model
        self.spatial_model = spatial_model
        self.source = source
        self.receiver = receiver
        self.a = torch.tensor([-2.5,4/3,-5/60]).to(device)
        self.seis = torch.zeros((self.time_model.nt,self.receiver.loc.shape[0])).to(device)
    def forward(self):
        # 中间参数
        alpha = torch.tensor((self.spatial_model.V * self.time_model.dt/self.spatial_model.spacing)**2).to(device)
        # beta = alpha*self.spatial_model.spacing**2
        P_t = torch.zeros(self.spatial_model.V.shape).to(device)
        P_t_1 = torch.zeros(self.spatial_model.V.shape).to(device)
        ricker_wave = torch.tensor(self.source.wave.ricker).to(device)
        for i in range(self.time_model.nt):
            P = alpha * (
                self.a[1]*(torch.roll(P_t,1,dims=0)+torch.roll(P_t,1,dims=1)+torch.roll(P_t,1,dims=2)+
                        torch.roll(P_t,-1,dims=0)+torch.roll(P_t,-1,dims=1)+torch.roll(P_t,-1,dims=2))+
                self.a[2]*(torch.roll(P_t,2,dims=0)+torch.roll(P_t,2,dims=1)+torch.roll(P_t,2,dims=2)+
                        torch.roll(P_t,-2,dims=0)+torch.roll(P_t,-2,dims=1)+torch.roll(P_t,-2,dims=2))
            ) + (alpha*3*self.a[0]+2)*P_t - P_t_1
            P[self.source.loc] -= self.spatial_model.spacing**2*alpha[self.source.loc] * ricker_wave[i]
            # save results
            for ir in range(self.receiver.loc.shape[0]):
                self.seis[i,ir] = P[tuple(self.receiver.loc[ir])]
            P_t_1 = P_t
            P_t = P
        del alpha
        del P_t
        del P_t_1
        del P
        del ricker_wave
        return self.seis.to('cpu')
