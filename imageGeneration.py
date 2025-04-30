import argparse
import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

from MicroStripArrangementEnv import MicroStripArrangementEnv as msaEnv
import MicroStripArrangementLib as msaLib

def plot(imgdir: str, plot: str, X: list, Y: list, labelX: str, labelY: str):
    imgFile = os.path.join(imgdir,f'{plot}.png')
    plt.figure(figsize=(15,10))
    plt.plot(X, Y, color = 'green',label='_nolabel')
   
    # plt.legend(loc='upper right')
    plt.ylabel(labelY)
    plt.xlabel(labelX)
    plt.grid(True)
    plt.savefig(imgFile)
    plt.close()

def string_to_array(s):
    # Remove brackets and split by whitespace
    return np.array([float(x) for x in s.strip('[]').split()], dtype=float)
   
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
        envCase: str):
    
    env = msaEnv(V0=V0,hw_arra=a*1e-3,ht_arra=b*1e-3,ht_subs=c*1e-3,hw_micrstr=d*1e-3,ht_micrstr=t,er1=er1,er2=er2,num_fs=num_fs,num_pts=num_pts)
    
    cwd = os.getcwd()
    actionFileName = os.path.join(cwd,"training",envCase,"images","SAC","action.csv")
    
    dfAction = pd.read_csv(actionFileName)
    dfAction['action'] = dfAction['action'].apply(string_to_array) # csv format stored the action values as a string without comma separators
    
    resultDict = {
      "timestepList" : [],
        "energyList" : [],
        "rewardList" : [],
        "controlPointsList" : [] , 
    }
    
    
    for i in range(len(dfAction)):
        action = dfAction['action'].iloc[i]
        timestep = dfAction['timestep'].iloc[i]
        resultDict["timestepList"].append(timestep)
        
        gptsX,gptsY,controlPoints = env.getBezierCurve(action)
        resultDict["controlPointsList"].append(controlPoints)
        
        vn = msaLib.calculatePotentialCoeffs(env.V0,env.hw_micrstr,env.hw_arra,env.num_fs,gptsY[1:-1],gptsX[1:-1])
        predEnergy = msaLib.calculateEnergy(env.er1,env.er2,env.hw_arra,env.ht_arra,env.ht_subs,env.num_fs,vn)
        resultDict["energyList"].append(predEnergy)
        
        resultDict["rewardList"].append(env.getReward(gptsX,gptsY,action))
    
    imgdir = os.path.join(cwd,"training",envCase,"images","SAC")  
    plot(imgdir, "Energy", resultDict["timestepList"], resultDict["energyList"], "Timestep", "Energy [VAs]")
    
    fileName = os.path.join(imgdir,"IntermediateResults.csv")
    (pd.DataFrame(resultDict).to_csv(fileName,index=False))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create plots")
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
    
        
    # run the script from terminal using
    # python imageGeneration.py -V0 1.0 -a 1.38 -b 2.76 -c 0.1382 -d 0.05 -t 0.0 -er1 1.0 -er2 12.9 -N 2000 -npts 53
    main(args.V0,args.a,args.b,args.c,args.d,args.t,args.er1,args.er2,args.N,args.npts,envCase)
        