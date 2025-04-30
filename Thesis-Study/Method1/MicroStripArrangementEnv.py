#################################################
# Author : Philip Varghese Modayil
# Date   : 29.05.2023
# Topic  : Microstrip Arrangement RL Environment
#################################################

# import libraries
import numpy as np

# import the custom MicroStripArrangement library
from MicroStripArrangement import MicroStripArrangement as msa

# import gym
from gym import Env
from gym.spaces import Box


# Custom environment for scheduling process
class MicroStripArrangementEnv(Env,msa):
    """
    This class initializes a micro strip arrangement RL environment
    """
    
    """
        ### Methodology
        -----------------------------------------
   
        
    """
    
    # initialization function
    def __init__(self):
        
        # initialize the Micro Strip Arrangement class
        msa.__init__(self,V0=1,hw_arra=3e-2,ht_arra=2e-2,ht_subs=1e-3,hw_micrstr=0.1e-3,ht_micrstr=0,er_1=1,er_2=4.8,num_fs=3000)
        
        # considering 10 points on the x-axis between hw_micrstr + delta to hw_arra - delta
        self.num_pts = 10
        self.delta_resolution = 10 # resolution for delta
        self.delta = (self.hw_arra - self.hw_micrstr)/self.delta_resolution # delta to keep the points away from the critical points hw_micrstr & hw_arra
        
        # x-axis points
        self.x = np.linspace(self.hw_micrstr + self.delta,self.hw_arra - self.delta,self.num_pts)
        
        # current energy of the state
        self.current_energy = 1000
        # minimum energy in lifetime
        self.minimum_energy = 1000
        
        # reward
        self.reward = 0
        
        # per episode take 10 steps
        self.iter_value = 10
        
        """
        Action Space
        -----------------------------------------
        | Action          | Min               | Max                | Size             |   
        |-----------------|-------------------|--------------------|------------------|
        | g points        | 0                 | 1                  | ndarray(num_pts,)|
        
        """
        self.action_space = Box(low=np.float32(0), high=np.float32(1),shape=(self.num_pts,), dtype=np.float32)
        
        """
        Observation Space
        -----------------------------------------
        The observation space includes the current g points(current action), hw_micrstr, hw_arra, minimum_energy, current_energy.
        
        The observation space is an `ndarray` with shape `(self.num_pts+4,)` where the elements correspond to the following:
        
        | Num           | Observation           | Min               | Max                |
        |---------------|-----------------------|-------------------|--------------------|
        | 0 - num_pts-1 | g points              | 0                 | 1                  |
        | +1            | hw_micrstr            | 0                 | Inf                |
        | +2            | hw_arra               | 0                 | Inf                |
        """
        self.observation_space = Box(low=0, high=np.inf,shape=(self.num_pts+2,), dtype=np.float32)
        
    # environment reset function
    def reset(self):
        
        # reset reward
        self.reward = 0
        
        # per episode take 10 steps
        self.iter_value = 10
        
        # pass auxilliary info
        info = {"Global_Minimum_Energy":self.minimum_energy, "Current_Minimum_Energy":self.current_energy}
        
        obs_space = np.ones(self.num_pts, dtype=np.float32)
        obs_space = np.append(obs_space,self.hw_micrstr)
        obs_space = np.append(obs_space,self.hw_arra)
        return obs_space
    
    # envirionment step function  
    def step(self, action):
        # set done as False 
        done = False
        
        # # decrement iter_value
        # self.iter_value -= 1
        
        # get the g points from RL agent
        g_pts = action
        
        """
        Reward
        -----------------------------------------
        
        | Within 0 and 1 | Monotone Decreasiing | Convex | Minimum Energy | Reward                  |
        |----------------|----------------------|--------|----------------|-------------------------|
        | True           | True                 | True   | True           | +1+2+2+1/minimum_energy |
        | True           | True                 | True   | False          | +1+2+2                  |
        | True           | True                 | False  | -              | +1+2-2                  |
        | True           | False                | -      | -              | +1-2                    |
        | False          | -                    | -      | -              | -1                      |
        
        """
        # all points should be within 0 and 1
        if np.all(g_pts < 1) and np.all(g_pts > 0):
            self.reward += 1
            
            # monotone decreasing
            if self.monotonically_decreasing(g_pts):
                #print('g points are monotonically decreasing')
                self.reward += 2
                
                # convex
                if self.is_convex(g_pts):
                    #print('g points are convex')
                    
                    self.reward += 2
                    # calculate potential coefficients
                    vn = self.potential_coeff(g_pts,self.x)
                    
                    # calculate energy
                    self.current_energy = self.energy(vn)
                    
                    if self.current_energy < self.minimum_energy:
                        self.minimum_energy = self.current_energy
                        self.reward += 1/self.current_energy
                # not convex
                else:
                    self.reward -= 2
            # not monotone decreasing
            else:
                #print('g points are not monotone decreasing')
                self.reward -= 2
        # not within 0 and 1
        else:
            self.reward -= 1
        
        # observation space
        obs_space = g_pts
        obs_space = np.append(obs_space,self.hw_micrstr)
        obs_space = np.append(obs_space,self.hw_arra)
        #print(obs_space)
        
        # # check for end of episode
        # if self.iter_value <=0:
        #     # set done to True
        done = True
        
        # pass auxilliary info
        info = {"Global_Minimum_Energy":self.minimum_energy, "Current_Minimum_Energy":self.current_energy}
        
        # Return step information
        return obs_space, self.reward, done,info

    def render(self):
        pass