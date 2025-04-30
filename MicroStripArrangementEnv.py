#################################################
# Author : Philip Varghese Modayil
# Date   : 21.10.2024
# Topic  : Microstrip Arrangement RL Environment
# Method : Cubic Bezier Curve, scaled rewards
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import numpy as np
from typing import Tuple
# import gym
from gymnasium import Env
from gymnasium.spaces import Box
# import the custom MicroStripArrangement library
import MicroStripArrangementLib as msaLib

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#####################################################################################
#                               Custom Gym Environment
#####################################################################################
class MicroStripArrangementEnv(Env):
    
    #####################################################################################
    #                               Initialisation
    #####################################################################################
    def __init__(self, 
                V0: float, 
                hw_arra: float,
                ht_arra: float,
                ht_subs: float,
                hw_micrstr: float,
                ht_micrstr: float,
                er1: float,
                er2: float,
                num_fs: int,
                num_pts: int) -> None:
        
        # Set up the microstrip
        ########################################################################
        self.V0 = V0 # potential of the microstrip
        self.hw_arra = hw_arra # half width of the arrangement
        self.ht_arra = ht_arra # height of the arrangement
        self.ht_subs = ht_subs # height of the substrate
        self.hw_micrstr = hw_micrstr # half width of the microstrip
        self.ht_micrstr = ht_micrstr # height of the microstrip
        self.er1 = er1 # relative permitivitty of medium 1(usually air)
        self.er2 = er2 # relative permitivitty of medium 2
        self.num_fs =num_fs # number of fourier coefficients(vn)
        self.num_pts = num_pts # number of g points to be generated for hw_micrstr <= x <= hw_arra
        
        # initial curve energy used to scale the reward in the reward function
        ##########################################################################
        initialCurveX, initialCurveY,_ = self.getBezierCurve(np.array([0,0,0,0]))
        vn = msaLib.calculatePotentialCoeffs(self.V0,self.hw_micrstr,self.hw_arra,self.num_fs,initialCurveY,initialCurveX)
        self.initialCurveEnergy = msaLib.calculateEnergy(self.er1,self.er2,self.hw_arra,self.ht_arra,self.ht_subs,self.num_fs,vn)
        self.minimumEnergy = np.array([np.inf]) # to keep track of latest minimum energy, first element is set as the highest possible value
        
        """
        Action Space
        -----------------------------------------
        The action space is an array of size four where two pairs of values specify the (x,y) coordinates of the two control points. 
            
        | Action          | Min               | Max                | Size       |   
        |-----------------|-------------------|--------------------|------------|
        | control fcator  | -1                | 1                  | ndarray(4,)|
        
        """
        # action_space variable is as in the gym Env
        self.action_space_boundary = 0.2
        self.action_space = Box(low=-self.action_space_boundary, high=self.action_space_boundary,shape=(4,), dtype=np.float32) 
        
        """
        Observation Space
        -----------------------------------------
        The observation space includes hw_micrstr, hw_arra, ht_arra, ht_subs and er2
        
        The observation space is an `ndarray` with shape `(5,)` where the elements correspond to the following:
        
        | Num          | Observation           | Min               | Max                |
        |--------------|-----------------------|-------------------|--------------------|
        | 0            | hw_micrstr            | 0                 | Inf                |
        | 1            | hw_arra               | 0                 | Inf                |
        | 2            | ht_arra               | 0                 | Inf                |
        | 3            | ht_subs               | 0                 | Inf                |
        | 4            | er2                   | 0                 | Inf                |
        """
        # observation_space variable is as in the gym Env 
        self.observation_space = Box(low=0, high=np.inf,shape=(5,), dtype=np.float32)
    
    #####################################################################################
    #                              Generate Curve
    #####################################################################################   
    def getBezierCurve(self,action:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        generate the bezier curve for the given action(control points)

        Parameters
        ----------
        action : np.ndarray
            action generated by the RL agent

        Returns
        -------
        Tuple[np.ndarray,np.ndarray,np.ndarray]
            curveX, curveY, controlPoints
        """
        # control points
        ##################################################################
        P0 = np.array([self.hw_micrstr, 1])
        P10 = self.hw_micrstr + action[0]*(self.hw_arra-self.hw_micrstr)
        P1 = np.array([P10, action[1]])
        P20 = self.hw_micrstr + action[2]*(self.hw_arra-self.hw_micrstr)
        P2 = np.array([P20, action[3]])
        P3 = np.array([self.hw_arra, 0])
        
        # bezier curve => 
        # (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
        ####################################################################
        # generate equidistant distribution
        tValues = np.linspace(0,1,self.num_pts) # 1Xnum_pts
        t = tValues[:, np.newaxis] # num_ptsX1
        
        # generate the bezier curve coefficients
        B0 = (1 - t)**3
        B1 = 3 * t * (1 - t)**2
        B2 = 3 * t**2 * (1 - t)
        B3 = t**3
        
        # stack the control points to do matrix multiplication
        controlPoints = np.array([P0, P1, P2, P3]) # 4X2
        curveCoefficients = np.column_stack((B0, B1, B2, B3)) # num_ptsX4
        # Calculate the curve points using matrix multiplication
        curvePoints = np.matmul(curveCoefficients, controlPoints) # num_ptsX2
        
        # Calculate the x and y coordinates using the quadratic Bézier formula
        gptsX = curvePoints[:, 0]
        gptsY = curvePoints[:, 1]
        
        return gptsX,gptsY,controlPoints
    
    #####################################################################################
    #                                Get Reward
    #####################################################################################
    def getReward(self, gptsX: np.ndarray, gptsY: np.ndarray, action: np.ndarray) -> float:
        """
        obtain reward value for the generated bezier curve

        Parameters
        ----------
        gptsX : np.ndarray
            curve x-coordinates
        gptsY : np.ndarray
            curve y-coordinates
        action : np.ndarray
            RL agent action array

        Returns
        -------
        reward: float
            reward value
        """
        # initialize
        ################
        reward = 0
        
        # to promote some change
        ######################################
        if np.all(action == 0):
            # conditon where no chnage happens
            return -100
        
        if np.any(action == self.action_space_boundary):
            # condition where it gets extreme limits
            return -100
        
        # is monotone decreasing and convex
        #######################################
        if msaLib.isMonotonicallyDecreasing(gptsY):
            # is convex
            #######################################
            if msaLib.isConvex(gptsY):
                
                # calculate potential coefficients
                #######################################
                vn = msaLib.calculatePotentialCoeffs(self.V0,self.hw_micrstr,self.hw_arra,self.num_fs,gptsY[1:-1],gptsX[1:-1])
                
                # calculate energy
                #######################################
                currentEnergy = msaLib.calculateEnergy(self.er1,self.er2,self.hw_arra,self.ht_arra,self.ht_subs,self.num_fs,vn)
                
                # update reward
                #######################################
                reward += 20 + ((1/currentEnergy)/(1/self.initialCurveEnergy))*1e2 # 20 to keep it above the reward of necessary conditions
                # new minimum energy achieved 
                #######################################   
                if currentEnergy <= self.minimumEnergy[-1]:
                    # update minimum energy
                    
                    self.minimumEnergy = np.append(self.minimumEnergy,currentEnergy)
                    logger.info(f"New minimum energy obtained.... New Minimum Energy: {self.minimumEnergy[-1]}")

                                   
            else:
                reward += msaLib.degreeOfMonotonicity(gptsY) + msaLib.degreeOfConvexity(gptsY)
                
        # not monotone 
        #######################################
        else:
            reward += msaLib.degreeOfMonotonicity(gptsY)
        
        # ensure the dtype of reward
        #######################################
        return float(reward)
    
    # environment reset function
    def reset(self, seed = None) -> Tuple[np.ndarray,dict]:
        """
        reset function according to gym Env, not very important for DDRL(Degenrate DRL) approach.
        
        Parameters
        ----------
        seed : _type_, optional
            not important in this approach since environment does not have randomness, by default None

        Returns
        -------
        Tuple[np.ndarray,dict]
            observation space, info
        """
        # observation space
        ######################################
        obs_space = np.array([self.hw_micrstr, self.hw_arra, self.ht_arra, self.ht_subs, self.er2])
        obs_space = obs_space.astype(np.float32) # ensure type
        
        info = {}
        return obs_space, info
    
    # envirionment step function  
    def step(self, action: np.ndarray) -> Tuple[np.ndarray,float,bool,bool,dict]:
        """
        one step of training

        Parameters
        ----------
        action : np.ndarray
            action array from RL agent

        Returns
        -------
        Tuple[np.ndarray,float,bool,bool,dict]
            observation space, reward, terminated flag, truncated flag, info
        """
        
        # set terminated as False
        ###################################################### 
        terminated = False
        truncated = False

        # get absolute value of action since the action distribution is symmetrical over zero
        action = np.abs(action)
        
        # get the g points 
        #######################################################
        gptsX,gptsY,controlPoints = self.getBezierCurve(action)
    
        # set rewards
        ##########################################################
        reward = self.getReward(gptsX,gptsY,action)
                
    
        # observation space
        #############################################################
        obs_space = np.array([self.hw_micrstr, self.hw_arra, self.ht_arra, self.ht_subs, self.er2])
        obs_space = obs_space.astype(np.float32) # ensure type
        
        # set done to True
        terminated = True
        
        # pass auxilliary info
        info = {}
        
        # Return step information
        return obs_space, reward, terminated, truncated, info

    def render(self):
        pass