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
from gymnasium import Env
from gymnasium.spaces import Box

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
    def __init__(self,V0,hw_arra,ht_arra,ht_subs,hw_micrstr,ht_micrstr,er_1,er_2,num_fs,num_pts):
        
        # initialize the Micro Strip Arrangement class (measurements are in mm scale)
        msa.__init__(self,V0,hw_arra,ht_arra,ht_subs,hw_micrstr,ht_micrstr,er_1,er_2,num_fs)
        
        # a delta for x axis points to stay away from boundary
        delta_resolution = 10 # resolution for delta
        delta = (self.hw_arra - self.hw_micrstr)/delta_resolution
        
        # x-axis points
        self.x = np.linspace(self.hw_micrstr + delta,self.hw_arra - delta,num_pts)
        
        # y-axis limits
        self.y = self.line_equ(self.hw_micrstr,self.hw_arra,self.x)
        
        # minimum energy in lifetime
        self.minimum_energy = np.inf
        
        # current energy
        self.current_energy = np.inf
        
        # minimum g_pts
        self.g_pts_min = np.ones(num_pts) 
        
        # reward
        self.reward = 0
        
        """
        Action Space
        -----------------------------------------
        The action space includes 5 control points which give the y-axis values for the following x-axis points
        (x_start + delta*0.125),(x_start + delta*0.25),(x_start + delta*0.375),(x_start + delta*0.5),(x_start + delta*0.875).
        these along with the (x,y) values of the boundaries make up the control points used for spline interpolation.
            
        | Action          | Min               | Max                | Size       |   
        |-----------------|-------------------|--------------------|------------|
        | control fcator  | 0                 | 1                  | ndarray(1,)|
        
        """
        self.action_space = Box(low=0, high=1,shape=(1,), dtype=np.float32)
        
        """
        Observation Space
        -----------------------------------------
        The observation space includes the current g points(current action), hw_micrstr, hw_arra, minimum_energy, current_energy.
        
        The observation space is an `ndarray` with shape `(4,)` where the elements correspond to the following:
        
        | Num          | Observation           | Min               | Max                |
        |--------------|-----------------------|-------------------|--------------------|
        | 0            | hw_micrstr            | 0                 | Inf                |
        | 1            | hw_arra               | 0                 | Inf                |
        | 2            | ht_arra               | 0                 | Inf                |
        | 3            | ht_subs               | 0                 | Inf                |
        | 4            | er_2                  | 0                 | Inf                |
        """
        self.observation_space = Box(low=0, high=np.inf,shape=(5,), dtype=np.float32)
    
    # function to return line equation parameters
    def line_equ(self,x_start,x_intercept,x):
        # the y-axis values are prior knowledge (x_start,1) (x_intercept,0)
        m = -1/(x_intercept-x_start)
        b = 0 - m*x_intercept
        
        # y = mx + b -> line equation
        y = m*x + b
        
        return y
    
    # function to return g points
    def e_func(self,x,action):
        # division by zero is not permitted
        if action[0] == 0:
            num_pts = len(x)
            return np.ones(num_pts)
         
        # e^(-(x-hw_micrstr)/action[0])
        g_pts = np.exp(-(x-self.hw_micrstr)/action[0])
       
        return g_pts
   
    # environment reset function
    def reset(self):
        
        # reset reward
        self.reward = 0
        
        # current energy
        self.current_energy = np.inf
        
        # observation space
        obs_space = self.hw_micrstr
        obs_space = np.append(obs_space,self.hw_arra)
        obs_space = np.append(obs_space,self.ht_arra)
        obs_space = np.append(obs_space,self.ht_subs)
        obs_space = np.append(obs_space,self.er_2)
        
        return obs_space
    
    # envirionment step function  
    def step(self, action):
        
        # set done as False 
        done = False
        
        # decrement iter_value
        # self.iter_value -= 1
        
        # get the g points from RL agent
        g_pts = self.e_func(self.x,action)
        
        """
        Reward
        -----------------------------------------
        
        | Within 0 and y | g_pts < g_pts_min | Minimum Energy | Reward                        |
        |----------------|-------------------|----------------|-------------------------------|
        | True           | True              | True           | 3*(1/current_energy)*0.000001 |
        | True           | False             | -              | 1*(1/current_energy)*0.000001 |
        | False          | -                 | -              | 0                             |
        """
        
        # all points should be within 0 and the line connecting x_start and x_intercept
        if np.all(g_pts < self.y) and np.all(g_pts > 0):
            # calculate potential coefficients
            vn = self.potential_coeff(g_pts,self.x)
            
            # calculate energy
            self.current_energy = self.energy(vn)
            
            # update reward
            self.reward += (1/self.current_energy)*0.000001
            
            # new minimum g_pts
            if np.all(g_pts < self.g_pts_min):
                # update minimum g_pts
                self.g_pts_min = g_pts
                
                # update reward
                self.reward += (1/self.current_energy)*0.000001
            
            # new minimum energy achieved    
            if self.current_energy < self.minimum_energy:
                # update minimum energy
                self.minimum_energy = self.current_energy
                
                # update reward
                self.reward += (1/self.current_energy)*0.000001
        
        # ensure the reward value is float
        self.reward = float(self.reward)
        
        # observation space
        obs_space = self.hw_micrstr
        obs_space = np.append(obs_space,self.hw_arra)
        obs_space = np.append(obs_space,self.ht_arra)
        obs_space = np.append(obs_space,self.ht_subs)
        obs_space = np.append(obs_space,self.er_2)
        
        # set done to True
        done = True
        
        # pass auxilliary info
        info = {"Global_Minimum_Energy":self.minimum_energy, 
                "Current_Minimum_Energy":self.current_energy, 
                "g_pts_min":self.g_pts_min,
                "y":self.y,
                "g_pts":g_pts, 
                "x":self.x}
        
        # Return step information
        return obs_space, self.reward, done,info

    def render(self):
        pass