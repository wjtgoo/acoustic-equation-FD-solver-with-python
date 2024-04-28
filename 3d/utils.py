import numpy as np

# Time term
class time_model:
    def __init__(self, dt:float,T:float):
        self.dt_ = dt
        self.T_ = T
    @property
    def dt(self):
        return self.dt_
    @property
    def T(self):
        return self.T_
    @property
    def nt(self):
        return int(self.T_//self.dt_)

# wavelet term
class wavelet:
    def __init__(self,time_model:time_model,freq:int=20):
        self.freq = freq
        self.dt = time_model.dt
        self.nt = time_model.nt
        self.time_model = time_model
        self.ricker_ = None
    @property
    def ricker(self):
        '''
        产生ricker子波
        ricker(t)=(1-2*(pi*f*t)^2)*exp(-(pi*f*t)^2)
        params:   
                freq:ricker子波的中心频率
                dt  :采样时间间隔 1/freq >> 2*dt(避免假频)
        '''
        # 判断是否会出现假频
        if 1/self.freq <= 2*self.dt:
            raise ValueError("ERROR! dissatisfy 1/freq >> 2*dt will get alias!")
        # 选取完整的波形
        # 2/freq表示完整波段周期，/dt来计算完整波段的时间步
        nt = 2/self.freq/self.dt
        nt = 2*np.floor(nt/2)+1# 取整
        period = np.floor(nt/2)# 选取中间值
        k = np.arange(1,nt+1)
        alpha = (period-k+1)*self.freq*self.dt*np.pi
        beta = alpha**2
        self.ricker_ = (1-beta*2)*np.exp(-beta)
        if self.time_model.nt <= len(self.ricker_):
            self.ricker_ =  self.ricker_[:nt]
        else:
            self.ricker_ = np.concatenate([self.ricker_,np.zeros(self.time_model.nt-len(self.ricker_))])
        return self.ricker_
# source term
class source:
    def __init__(self,time_model:time_model,location:tuple,freq:int=20):
        '''
        describe:
            only support one shot for now
        params:
            wave    : wavelet object
            location: grid number with shape (x,y,z)
        '''
        self.wavelet = wavelet(time_model,freq)
        self.location = location
    @property
    def wave(self):
        return self.wavelet
    @property
    def loc(self):
        return self.location
    
# receiver term
class receiver:
    def __init__(self,location:np.ndarray):
        '''
        params:
            location: grid number with shape [[x_1,y_1,z_1],[x_2,y_2,z_2],...,[x_n,y_n,z_n]]
        '''
        self.loc_ = location
    @property
    def loc(self):
        return self.loc_

# Spatial term
class spatial_model:
    def __init__(self, vel, spacing:float,nabc:int=0,abc_method:str='pml'):
        self.vel = vel # 可优化
        self.nabc = nabc
        self.abc_method = abc_method
        self.abc_vel = self.vel
        self.spacing_ = spacing
        self.npts_ = vel.shape
    def abc(self):
        if self.abc_method == 'pml':
            '''
            PASS
            '''
            self.abc_vel = self.vel
        else:
            raise ValueError("No such abc method")
    
    # 定义最终返回的速度图
    @property
    def V(self):
        return self.abc_vel
    @property
    def spacing(self):
        return self.spacing_
    @property
    def npts(self):
        return self.npts_