#################################################
# Author : Philip Varghese Modayil
# Date   : 26.01.2024
# Topic  : Microstrip Arrangement RL Environment
# Method : Cubic Bezier Curve
#################################################

# import libraries
import numpy as np
from typing import *

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
        Cubic Bezier Curve, Threshold Reward
        
    """
    
    # initialization function
    def __init__(self,V0,hw_arra,ht_arra,ht_subs,hw_micrstr,ht_micrstr,er_1,er_2,num_fs,num_pts) -> None:
        
        # initialize the Micro Strip Arrangement class (measurements are in mm scale)
        msa.__init__(self,V0,hw_arra,ht_arra,ht_subs,hw_micrstr,ht_micrstr,er_1,er_2,num_fs)
        
        action_space_size = 4
        self.num_pts = num_pts
        
        # minimum energy in lifetime
        self.minimum_energy = np.inf

        
        # x-axis values
        self.x = np.linspace(self.hw_micrstr ,self.hw_arra,num_pts)
        
        # line threshold y-axis values
        self.line_threshold = self.line_equ(self.x)
        
        # starting g points
        self.g_pts_start_x,self.g_pts_start_y,control_points = self.g_points_cubic_bezier(np.array([0,0,0,0]))
        self.energy_init = self.energy(self.potential_coeff_parallel(self.g_pts_start_y[1:-1],self.g_pts_start_x[1:-1]))
        
        # current energy
        self.current_energy = np.inf
        
        # reward
        self.reward = 0
        
        """
        Action Space
        -----------------------------------------
            
        | Action          | Min               | Max                | Size       |   
        |-----------------|-------------------|--------------------|------------|
        | control fcator  | -1                | 1                  | ndarray(1,)|
        
        """
        self.action_space = Box(low=-0.2, high=0.2,shape=(action_space_size,), dtype=np.float32)
        
        """
        Observation Space
        -----------------------------------------
        The observation space includes the current g points(current action), hw_micrstr, hw_arra, minimum_energy, current_energy.
        
        The observation space is an `ndarray` with shape `(5,)` where the elements correspond to the following:
        
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
    def line_equ(self,x:np.ndarray) -> np.ndarray:
        # set end point, slope and intercept
        ##########################################
        x_start = self.hw_micrstr
        x_intercept = self.hw_arra
        # the y-axis values are prior knowledge (x_start,1) (x_intercept,0)
        m = -1/(x_intercept-x_start)
        b = 0 - m*x_intercept
        
        # y = mx + b -> line equation
        ##########################################
        y = m*x + b
        
        return y
    
    def cubic_bezier_curve(self,t:np.ndarray,P0:np.ndarray,P1:np.ndarray,P2:np.ndarray,P3:np.ndarray) -> np.ndarray:
        # return the cubic bezier curve point for one (x,y) control points pair
        return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
    
    # function to return g points
    def g_points_cubic_bezier(self,action:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        # control points
        ##################################################################
        P0 = np.array([self.hw_micrstr, 1])
        P10 = self.hw_micrstr + action[0]*(self.hw_arra-self.hw_micrstr)
        P1 = np.array([P10, action[1]])
        P20 = self.hw_micrstr + action[2]*(self.hw_arra-self.hw_micrstr)
        P2 = np.array([P20, action[3]])
        P3 = np.array([self.hw_arra, 0])
        
        # Generate exponential distribution
        ###################################################################
        # t_temp = np.exp(np.linspace(-5, 0, self.num_pts))
        # # Normalize to the range [0, 1]
        # t_values = (t_temp - np.min(t_temp)) / (np.max(t_temp) - np.min(t_temp))
        
        # Generate equidistant distribution
        ###################################################################
        t_values = np.linspace(0,1,self.num_pts)
        
        # bezier curve
        ####################################################################
        curve_points = np.array([self.cubic_bezier_curve(t, P0, P1, P2, P3) for t in t_values])
        # Calculate the x and y coordinates using the quadratic Bézier formula
        g_ptsx = curve_points[:, 0]
        g_ptsy = curve_points[:, 1]
        
        control_points = np.array([P0,P1,P2,P3])    
        return g_ptsx,g_ptsy,control_points
   
    # environment reset function
    def reset(self, seed = None) -> Tuple[np.ndarray,dict]:
        
        # reset reward
        ######################################
        self.reward = 0
        
        # current energy
        ######################################
        self.current_energy = np.inf
        
        # observation space
        ######################################
        obs_space = self.hw_micrstr
        obs_space = np.append(obs_space,self.hw_arra)
        obs_space = np.append(obs_space,self.ht_arra)
        obs_space = np.append(obs_space,self.ht_subs)
        obs_space = np.append(obs_space,self.er_2)
        obs_space = obs_space.astype(np.float32)
        
        info = {}
        return obs_space, info
    
    def reward_calc(self,g_ptsx:np.ndarray,g_ptsy:np.ndarray,action:np.ndarray) -> float:
        """
        Reward
        -----------------------------------------
        reward:
        => not monotone decreasing:  reward -> reward + degree_of_monotinicity
        =>  monotone decreasing: reward ->  reward -> reward + 1/energy
                =>  energy < current minimum: reward -> reward + 1/energy
       """

        # initialize
        ###########################################################################
        reward = 0
        
        # to promote some change
        ###########################################################################
        if np.all(action == 0):
            # conditon where no chnage happens
            return -100
        
        if np.any(action == 0.2):
            # condition where it gets stuck at line_threshold_upper 
            return -100
        
        # is monotone decreasing and convex
        ###########################################################################
        if self.monotonically_decreasing(g_ptsy):
            # is convex
            ##############################################################
            if self.is_convex(g_ptsy):
                
                # calculate potential coefficients
                ########################################################
                vn = self.potential_coeff_parallel(g_ptsy[1:-1],g_ptsx[1:-1])
                
                # calculate energy
                #########################################################
                self.current_energy = self.energy(vn)
                
                # update reward
                ########################################################
                reward += 20 + ((1/self.current_energy)/(1/self.energy_init))*1e2 # 20 to keep it above the reward of necessary conditions
                # new minimum energy achieved 
                #######################################################   
                if self.current_energy <= self.minimum_energy:
                    # update minimum energy
                    self.minimum_energy = self.current_energy
                    
                    # update minimum energy action
                    self.action_minimum_energy = action

                    # # update reward
                    # reward += (1/self.current_energy)*1e-5
                                   
            else:
                reward += self.degree_of_monotonicity(g_ptsy) + self.degree_of_convex(g_ptsy)
                
        # not monotone 
        ##################################################################################
        else:
            reward += self.degree_of_monotonicity(g_ptsy)
        
        # ensure the dtype of reward
        #####################################################################################################################
        reward = float(reward) 
        
        return reward
    
    # envirionment step function  
    def step(self, action) -> Tuple[np.ndarray,float,bool,bool,dict]:
        
        # set terminated as False
        ###################################################### 
        terminated = False
        truncated = False

        action = np.abs(action)
        # get the g points from RL agent
        #######################################################
        g_ptsx,g_ptsy,control_points = self.g_points_cubic_bezier(action)
    
        # set rewards
        ##########################################################
        self.reward = self.reward_calc(g_ptsx,g_ptsy,action)
                
    
        # observation space
        #############################################################
        obs_space = self.hw_micrstr
        obs_space = np.append(obs_space,self.hw_arra)
        obs_space = np.append(obs_space,self.ht_arra)
        obs_space = np.append(obs_space,self.ht_subs)
        obs_space = np.append(obs_space,self.er_2)
        obs_space = obs_space.astype(np.float32)
        
        # set done to True
        terminated = True
        
        # pass auxilliary info
        info = {}
        
        # Return step information
        return obs_space, self.reward, terminated,truncated,info

    def render(self):
        pass