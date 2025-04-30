#################################################
# Author : Philip Varghese Modayil
# Date   : 21.10.2024
# Topic  : Microstrip Arrangement RL Agent Training
# Method : Cubic Bezier Curve, scaled rewards, SAC Agent
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import argparse
import torch
import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from importlib import reload

import MicroStripArrangementLib as msaLib
from MicroStripArrangementEnv import MicroStripArrangementEnv as msaEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

reload(msaLib)

#####################################################################################
#                                     Functions
#####################################################################################
# cretae directories
###############################################################
def create_directories(**kwargs):
    """
    takes in n number of directory paths and creates directories
    """
    for dir_name in kwargs.values():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")
        else:
            print(f"Directory already exists: {dir_name}")

def predict(env: msaEnv, model: SAC) -> np.ndarray:
    """
    SAC model prediction returns the absolute value of the action(since action is sampled symmetrically over zero)

    Parameters
    ----------
    env : msaEnv
        microstrip arrangement environment
    model : SAC
        SAC model from stable-baselines3

    Returns
    -------
    action : np.ndarray
        absolute value of predicted action
    """
    obs_space,_ss = env.reset()
    action, _states = model.predict(obs_space,deterministic=True)
   
    return np.abs(action)

def plot(imgdir: str, nomCurveX: np.ndarray, nomCurveY: np.ndarray, gptsX: np.ndarray, gptsY: np.ndarray, controlPoints: np.ndarray, timestep: int) -> None:
    
    imgFile = os.path.join(imgdir,f'predictionResult_{timestep}.png')
    plt.figure(figsize=(15,10))
    plt.plot(nomCurveX*1000, nomCurveY, color = 'blue',label='Nominal')
    plt.plot(gptsX*1000, gptsY, color = 'green',label='Prediction')
    plt.scatter(controlPoints[:,0]*1000,controlPoints[:,1],color='red',label='Control Points')
    plt.legend(loc='upper right')
    plt.ylabel('g points V(x,y=c) [Volt]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.savefig(imgFile)
    plt.close()

#####################################################################################
#                                     Training
##################################################################################### 
class IntermediatePredictionCallback(BaseCallback):
    """
    A custom callback that derives from `BaseCallback`
    Does intermediate prediction and generates images

    """
    def __init__(self, env: msaEnv, output_interval: int, imgdir: str, nomCurveX: np.ndarray, nomCurveY: np.ndarray, envCase: str):
        super(IntermediatePredictionCallback, self).__init__()
        self.env = env
        self.output_interval = output_interval
        self.imgdir = imgdir
        self.nomCurveX = nomCurveX
        self.nomCurveY = nomCurveY
        self.envCase = envCase
        self.action_df = {'action':[],
                          'timestep':[]}
        self.saveActionFile = os.path.join(imgdir,'action.csv')
        self.entropy_coefficient_set = False

    def _on_step(self) -> bool:
        if self.num_timesteps == 1:
            # initial prediction
            ############################################
            logger.info("Initial prediction......")
            action = predict(self.env,self.model)
            # gptsX,gptsY,controlPoints = self.env.getBezierCurve(action)
            # plot(self.imgdir, self.nomCurveX, self.nomCurveY, gptsX, gptsY, controlPoints, self.num_timesteps)

            self.action_df['action'].append(action)
            self.action_df['timestep'].append(self.num_timesteps)
            (pd.DataFrame(self.action_df)).to_csv(self.saveActionFile,index=False)
            logger.info("Resuming training......")
            
        # Check if the current timestep is a multiple of the output interval
        if self.num_timesteps % self.output_interval == 0:
            logger.info(f"Intermediate prediction at timestep: {self.num_timesteps}......")
            # Function to predict and generate output images
            action = predict(self.env,self.model)
            # gptsX,gptsY,controlPoints = self.env.getBezierCurve(action)
            # plot(self.imgdir, self.nomCurveX, self.nomCurveY, gptsX, gptsY, controlPoints, self.num_timesteps)

            self.action_df['action'].append(action)
            self.action_df['timestep'].append(self.num_timesteps)
            (pd.DataFrame(self.action_df)).to_csv(self.saveActionFile,index=False)
            logger.info("Resuming training......")
        
        return True

def train(env: msaEnv, 
    TIMESTEPS: int, 
    intermediatePredictionInterval: int, 
    modelDirRoot: str, 
    logDirRoot: str, 
    imageDirRoot: str, 
    envCase: str, 
    nomCurveX: np.ndarray, 
    nomCurveY: np.ndarray) -> None:
    """
    train _summary_

    _extended_summary_

    Parameters
    ----------
    env : msaEnv
        microstrip arrangement environment class object (inherited from gymmnasium.Env)
    TIMESTEPS : int
        total training timesteps
    intermediatePredictionInterval : int
        interval for intermediate results, must be less than TIMESTEPS
    modelDirRoot : str
        root directory for models
    logDirRoot : str
        root directory for log
    imageDirRoot : str
        root directory for images
    envCase : str
        CaseL/CaseD
    nomCurveX : np.ndarray
        nominal curve x-coordinates
    nomCurveY : np.ndarray
        nominal curve y-coordinates

    Returns
    -------
    None
        saves training stats
    """
    # create directories specific to case and agent
    ##############################################
    modelsDir = os.path.join(modelDirRoot,'SAC')
    logDir = os.path.join(logDirRoot,'SAC')
    imgDir = os.path.join(imageDirRoot,'SAC')
    create_directories(mdir = modelsDir, ldir = logDir, idir = imgDir)
    
    # define the SAC model and hyperparameters
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300]) # setting nn architecture
    hyperparams = {
        'ent_coef': 'auto',  # setting entropy coefficient to auto boosts exploration
        'batch_size':256,  
        'buffer_size': 20000,        
        'learning_rate': 0.0007,               
        'gamma': 0.98,      
        'tau':0.02,
        'learning_starts':1000,
        'use_sde':True      
    }
    
    model = SAC("MlpPolicy", 
                env,  
                verbose=0, 
                policy_kwargs=policy_kwargs,
                tensorboard_log=logDir,
                device=device,
                **hyperparams)  
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}") # log output the model device
    model_device = next(model.policy.parameters()).device
    logger.info(f"Model device: {model_device}")
    
    # training
    ############################################
    logger.info("Training started......")
    startTime = time.time()
    model.learn(total_timesteps=TIMESTEPS,
        log_interval=4,
        reset_num_timesteps=True, 
        tb_log_name=envCase,
        progress_bar=True,
        callback=IntermediatePredictionCallback(env,intermediatePredictionInterval,imgDir,nomCurveX,nomCurveY,envCase)) #
    trainingTime = (time.time() - startTime)/3600 # in hrs
    logger.info(f"Training ended with total training time: {trainingTime}......")
    
    modelFile = os.path.join(modelsDir,f'MSAEnvSACModel{envCase}')
    model.save(modelFile,include="all")
    logger.info("Model saved......")
    
    # prediction
    ############################################
    logger.info("Predicting......")
    vn = msaLib.calculatePotentialCoeffs(env.V0,env.hw_micrstr,env.hw_arra,env.num_fs,nomCurveY[2:-1],nomCurveX[2:-1])
    nominalEnergy = msaLib.calculateEnergy(env.er1,env.er2,env.hw_arra,env.ht_arra,env.ht_subs,env.num_fs,vn)
    
    action = predict(env,model)
    gptsX,gptsY,controlPoints = env.getBezierCurve(action)
    vn = msaLib.calculatePotentialCoeffs(env.V0,env.hw_micrstr,env.hw_arra,env.num_fs,gptsY[1:-1],gptsX[1:-1])
    predEnergy = msaLib.calculateEnergy(env.er1,env.er2,env.hw_arra,env.ht_arra,env.ht_subs,env.num_fs,vn)  
    gptsX = np.insert(gptsX,0,0) # put the values at x = 0
    gptsY = np.insert(gptsY,0,1)
    curveDF = {
        'g_ptsx': gptsX,
        'g_ptsy': gptsY
    }
    saveCurveFile = os.path.join(os.getcwd(),"training",envCase,"result_curve.csv") # save the resultant curve into a csv file
    (pd.DataFrame(curveDF)).to_csv(saveCurveFile,index=False)
    
    training_stats_data = {
       "training_stats":{
           "prediction_energy": predEnergy,
           "nominal_energy": nominalEnergy,
           "training_time[hrs]": trainingTime,
           "minimum_energy_hist": env.minimumEnergy[1:].tolist(), # numpy arrays are not JSON searializable
       } ,
       
       "curve":{
           "action": action.tolist(),
       }
    }
    
    file_path = os.path.join(os.getcwd(),"training",envCase,"training_stats.json")
    with open(file_path, 'w') as json_file:
        json.dump(training_stats_data, json_file, indent=4)
    logger.info("Resultant curve and training stats saved......")
    
#####################################################################################
#                                   Main Function
#####################################################################################            
def main(V0: float, 
        a: float,
        b: float,
        c: float,
        d: float,
        t: float,
        er1: float,
        er2: float,
        num_fs: int,
        num_pts: int,
        envCase: str,
        TIMESTEPS: int,
        intermediatePredictionInterval: int) -> None:
    # create folders to save models and log files
    #############################################
    cwd = os.getcwd()  
    modelDirRoot = os.path.join(cwd,"training",envCase,"models")
    logDirRoot = os.path.join(cwd,"training",envCase,"logs")
    imageDirRoot = os.path.join(cwd,"training",envCase,"images")
    create_directories(mdirRoot = modelDirRoot, ldirRoot = logDirRoot, idirRoot = imageDirRoot)
    
    # read nominal data
    #############################################
    try:
        nominalCurveDir = os.path.join(cwd,'nominalCurve')
        if not os.path.exists(nominalCurveDir):
            raise FileNotFoundError(f"The directory {nominalCurveDir} does not exist")
    except FileNotFoundError:
        logger.error(f"Directory: {nominalCurveDir} does not exist")
        os.makedirs(nominalCurveDir)
        logger.info(f"Created directory: {nominalCurveDir}. Restart training with nominal curves in this directory")
        sys.exit(1)
    
    match envCase:
        case 'CaseD':
            dfNominal = pd.read_csv(os.path.join(nominalCurveDir,'Yamashita_NominalGPtsCaseDCase2_1.csv'))
        case 'CaseL':
            dfNominal = pd.read_csv(os.path.join(nominalCurveDir,'Yamashita_NominalGPtsCaseLCase2_1.csv'))

    nomCurveX = np.array(dfNominal['g_ptsx']) 
    nomCurveY = np.array(dfNominal['g_ptsy'])
    
    # training
    ###############################################
    # set microstrip arrangement environment, all environment values should be in SI units
    env = msaEnv(V0=V0,hw_arra=a*1e-3,ht_arra=b*1e-3,ht_subs=c*1e-3,hw_micrstr=d*1e-3,ht_micrstr=t,er1=er1,er2=er2,num_fs=num_fs,num_pts=num_pts)

    train(env,TIMESTEPS,intermediatePredictionInterval,modelDirRoot,logDirRoot,imageDirRoot,envCase,nomCurveX,nomCurveY)
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="SAC agent training script for MicroStripArrangement environment")
    parser.add_argument('-tim', '--TIMESTEPS', type=int,  dest='tim',default=50_000, help='total training timesteps')
    parser.add_argument('-intval', '--pred_interval',  dest='intval',type=int, default=5000, help='interval for intermediate prediction results')
    parser.add_argument('-V0', '--scaled_potential',  dest='V0',type=float, default=1.0, help='scaled voltage at the microstrip')
    parser.add_argument('-a', '--hw_arra',  dest='a',type=float, default=1.38, help='half width of the microstrip arrangement')
    parser.add_argument('-b', '--ht_arra',  dest='b',type=float, default=2.76, help='height of the microstrip arrangement')
    parser.add_argument('-c', '--ht_subs',  dest='c',type=float, default=0.1382, help='height of the substrate')
    parser.add_argument('-d', '--hw_micrstr',  dest='d',type=float, default=0.05, help='half width of the microstrip')
    parser.add_argument('-t', '--ht_micrstr',  dest='t',type=float, default=0.0, help='height of the microstrip')
    parser.add_argument('-er1', '--epsilonr_1',  dest='er1',type=float, default=1.0, help='electric permitivity of medium 1')
    parser.add_argument('-er2', '--epsilonr_2',  dest='er2',type=float, default=12.9, help='electric permitivity of medium 2')
    parser.add_argument('-N', '--num_fs',  dest='N',type=int, default=20, help='number of Fourier coefficients')
    parser.add_argument('-npts', '--num_pts',  dest='npts',type=int, default=10, help='number of g-points')
    args = parser.parse_args()
    
    match args.er2:
        case 1: 
            envCase = 'CaseL'
        case _:
            envCase = 'CaseD'
    
    logger.info(f"Starting SAC RL agent training for {envCase} Single Microstrip Arrangement......")        
    # run the script from terminal using
    # python agentTrain.py -tim 50000 -intval 5000 -V0 1.0 -a 1.38 -b 2.76 -c 0.1382 -d 0.05 -t 0.0 -er1 1.0 -er2 12.9 -N 2000 -npts 30
    main(args.V0,args.a,args.b,args.c,args.d,args.t,args.er1,args.er2,args.N,args.npts,envCase,args.tim,args.intval)