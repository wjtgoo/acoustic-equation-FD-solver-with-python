'''
2D 2-8 FD Acoustic Modeling

Date    :2023.4.17
Version : 1.0
Author  :WangJuntao
Cited by:https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html
'''
import os
import time
import numpy as np

class Wave_Forward:
    def __init__(self,
                 velocity_model,
                 x_grid_nums,
                 z_grid_nums,
                 dx,
                 bgnums,
                 nt,
                 dt,
                 freq,
                 s_loc,
                 r_loc,
                 record_gap=20,
                 save_seis='seismic.npy',
                 save_wavefield='wavefield.npy'
                 ):
        '''
        params:
            velocity_model: 速度图模型  array-like
            x_grid_nums   : x轴网格数量 int
            z_grid_nums   : z轴网格数量 int
            dx            : 网格的间隔(m) float/int dx = dz
            bgnums        : 吸收边界的网格数    int
            nt            : 时间步  int
            dt            : 采样时间间隔(s)  1/freq >> 2*dt(避免假频)   float
            freq          : ricker子波的中心频率(HZ) float/int
            s_loc         : source炮的位置(m)  tuple   (x,z)
            r_loc         : receiver的位置(m)   [(x1,z1),(x2,z2),...,(xn,zn)]
            record_gap    : 保存波场的时间步间隔(default:20)
            save_seis     : 用来保存地震图的文件名，默认在./output路径下
            save_wavefield: 用来保存波场图的文件名，默认在./output路径下
        '''
        self.velocity_model = velocity_model
        self.x_grid_nums = x_grid_nums
        self.z_grid_nums = z_grid_nums
        self.dx = dx
        self.dz = dx
        self.bgnums = bgnums
        self.nt = nt
        self.dt = dt
        self.freq = freq
        self.s_loc = s_loc
        self.r_loc = r_loc
        self.seis = None # 地震图
        self.wavefield = [] # 波场图
        self.record_gap = record_gap
        self.save_seis = save_seis
        self.save_wavefield = save_wavefield
    def forward(self):
        '''
        正演函数
        return: 
            seis: 地震波图 array-like (nt,len(r_loc))
            wavefield:  波场快照
        '''
        ## init
        # create output folder
        if not os.path.exists('./output'):
            os.makedirs('./output')
        # 接收器数量
        receiver_nums = len(self.r_loc)
        # 初始化地震图矩阵
        self.seis = np.zeros((self.nt,receiver_nums),dtype=np.float64)
        # 拓展后的速度图
        pad_vel = self.pad_velocity()
        # 吸收边界条件
        abc = self.ABC_coef(pad_vel)
        kappa = abc*self.dt
        # ricker wavelet
        ricker = self.ricker_wavelet()
        # 炮与接收器的索引位置
        isx,isz,irx,irz = self._adjust_loc()
        # 差分系数 c_{-i}=c_{i}
        c0=-205.0/72.0;c1=8.0/5.0;c2=-1.0/5.0;c3=8.0/315.0;c4=-1.0/560.0
        # 初始化$p^{t}_{x,z}$与$p^{t-1}_{x,z}$
        p_t = np.zeros_like(pad_vel,dtype=np.float64)
        p_t_1 = np.zeros_like(pad_vel,dtype=np.float64)
        # 对应$\alpha=\left(\frac{\Delta t \cdot v}{\Delta x}\right)^2$
        alpha = (pad_vel*self.dt/self.dx)**2
        temp1 = (2+2*c0*alpha-kappa)
        temp2 = (1-kappa)
        beta_dt = (pad_vel*self.dt)**2
        print('start forward.....')
        start = time.time()
        ## start iterate
        for it in range(self.nt):
            p = temp1*p_t-temp2*p_t_1+alpha*\
                (
                c1*(np.roll(p_t,1,axis=1)+np.roll(p_t,-1,axis=1)+np.roll(p_t,1,axis=0)+np.roll(p_t,-1,axis=0))+\
                c2*(np.roll(p_t,2,axis=1)+np.roll(p_t,-2,axis=1)+np.roll(p_t,2,axis=0)+np.roll(p_t,-2,axis=0))+\
                c3*(np.roll(p_t,3,axis=1)+np.roll(p_t,-3,axis=1)+np.roll(p_t,3,axis=0)+np.roll(p_t,-3,axis=0))+\
                c4*(np.roll(p_t,4,axis=1)+np.roll(p_t,-4,axis=1)+np.roll(p_t,4,axis=0)+np.roll(p_t,-4,axis=0))
                )
            p[isz,isx] = p[isz,isx] + beta_dt[isz,isx]*ricker[it]
            # wavefield snapshot
            # 波场保存在output文件夹中
            if it % self.record_gap==0:
                wavefield = p[self.bgnums:self.bgnums+self.z_grid_nums,self.bgnums:self.bgnums+self.x_grid_nums]
                self.wavefield.append(wavefield)
            # Seismic Data
            for ir in range(len(self.r_loc)):
                self.seis[it,ir] = p[irz[ir],irx[ir]]
            # update
            p_t_1 = p_t
            p_t = p
        end = time.time()
        print('Finished with {:.2f}s'.format(end-start))
        self.wavefield = np.array(self.wavefield)
        np.save('./output/'+self.save_wavefield,self.wavefield)
        np.save('./output/'+self.save_seis,self.seis)
        print('Results saved in ./output')
    def ABC_coef(self,pad_vel):
        '''
        吸收边界条件
        params:
            pad_vel:扩展后的速度图(利用pad_velocity函数产生)
        return:
            damp: 吸收边界  array-like  damp.shape=pad_vel.shape
                  在吸收边界上值离边界距离递增,在其他区域值为0
        '''
        nz_pad,nx_pad = pad_vel.shape
        vel_min = pad_vel.min()
        nz = nz_pad-2*self.bgnums
        nx = nx_pad-2*self.bgnums
        L = (self.bgnums-1)*self.dx
        k = 3.0*vel_min*np.log(1e7)/(2.0*L)
        damp1d = k*(np.arange(self.bgnums)*self.dx/L)**2
        damp = np.zeros((nz_pad,nx_pad))
        for iz in range(nz_pad):
            damp[iz,:self.bgnums] = damp1d[::-1]
            damp[iz,nx+self.bgnums:nx+2*self.bgnums] = damp1d
        for ix in range(self.bgnums,self.bgnums+nx):
            damp[:self.bgnums,ix] = damp1d[::-1]
            damp[self.bgnums+nz:2*self.bgnums+nz,ix] = damp1d
        return damp
    def pad_velocity(self):
        '''
        扩展速度图边界
        return:
            pad_vel 扩展边界后的速度图
            
        example:
            vel = [[1,2,3],
                   [4,5,6],
                   [7,8,9]]
            bgnums = 2
            
            ==>
            
            pad_vel = [[1, 1, 1, 2, 3, 3, 3],
                       [1, 1, 1, 2, 3, 3, 3],
                       [1, 1, 1, 2, 3, 3, 3],
                       [4, 4, 4, 5, 6, 6, 6],
                       [7, 7, 7, 8, 9, 9, 9],
                       [7, 7, 7, 8, 9, 9, 9],
                       [7, 7, 7, 8, 9, 9, 9]]
        '''
        v1 = np.tile(self.velocity_model[:,0].reshape(-1,1),[1,self.bgnums])
        v2 = np.tile(self.velocity_model[:,-1].reshape(-1,1),[1,self.bgnums])
        pad_vel = np.concatenate((v1,self.velocity_model,v2),axis=1)
        v1 = np.tile(pad_vel[0,:].reshape(1,-1),[self.bgnums,1])
        v2 = np.tile(pad_vel[-1,:].reshape(1,-1),[self.bgnums,1])
        pad_vel = np.concatenate((v1,pad_vel,v2),axis=0)
        return pad_vel
    
    def ricker_wavelet(self):
        '''
        产生ricker子波

        params:   
                freq:ricker子波的中心频率
                dt  :采样时间间隔 1/freq >> 2*dt(避免假频)
        '''
        # 2.2除以freq是为了完整的采样到波动段
        nt = 2./self.freq/self.dt
        nt = 2*np.floor(nt/2)+1
        period = np.floor(nt/2)
        k = np.arange(1,nt+1)
        alpha = (period-k+1)*self.freq*self.dt*np.pi
        beta = alpha**2
        ricker = (1-beta*2)*np.exp(-beta)
        if len(ricker) < self.nt:
            ricker = np.concatenate([ricker,np.zeros(self.nt-len(ricker))])
        return ricker
    def _adjust_loc(self):
        '''
        由于添加了吸收边界层，接收器与炮的绝对位置发生了变化
        return:
            isx:    炮在扩充边界后的x轴索引位置
            isz:    炮在扩充边界后的z轴索引位置
            irx:    接收器在扩充边界后的x轴索引位置
            irz:    接收器在扩充边界后的z轴索引位置
        '''
        isx = int(self.s_loc[0]/self.dx)+self.bgnums
        isz = int(self.s_loc[1]/self.dz)+self.bgnums
        irx = [int(self.r_loc[i][0]/self.dx)+self.bgnums for i in range(len(self.r_loc))]
        irz = [int(self.r_loc[i][1]/self.dz)+self.bgnums for i in range(len(self.r_loc))]
        return (isx,isz,irx,irz)