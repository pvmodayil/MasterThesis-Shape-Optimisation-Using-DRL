"""
Genetic Algorithm Optimization of RL Predicted Curve
---------------------------------------------------------------------
    This program further optimizes the RL predicted with Gentic Algorithm
    Input: Starting Curve from RL Prediction
    Output: GA Optimized Curve
    
    Microstrip Arrangement
    ----------------------
    a: half width microstrip arrangement
    d: half width microstrip
    b: height of arrangement
    
    Note
    -----------------------
    values at x = 0 , x = d , and x = a are fixed and the input curve must not have value between x = 0 and x = d. 
    Otherwise it affects the size of the curve and thereby affects the program.
"""

# import
#------------------------------------------
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# import GA
#-----------------------------------------------
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import gentic_algorithm_optimizer as GAopt
from importlib import reload
reload(GAopt)

# initialize the best solution
#----------------------------------------------
best_solutions = []

# function to plot initial curve
#--------------------------------------------------------------------------------------
def plot_initial_curve(x_i,df_rlPred,df_nom,upper,x_threshold,y_threshold,init_imgfile):
    
    fig = plt.figure(figsize=(15,10))
    plt.plot(df_rlPred['g_ptsx']*1000,df_rlPred['g_ptsy'], color = 'green',label="RL Prediction")
    plt.plot(df_nom['g_ptsx']*1000,df_nom['g_ptsy'], color = 'blue', label="Nominal")
    plt.plot(x_threshold*1000,y_threshold, color = 'red', label="Threshold")

    plt.legend(loc='upper right',fontsize=20)
    plt.ylabel('g points V(x,y=c) [Volt]',fontsize=20)
    plt.xlabel('x axis [mm]',fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.savefig(f"{init_imgfile}")
    plt.close()

# function to plot result curve
#--------------------------------------------------------------------------------------    
def plot_result(x_i,df_rlPred,df_nom,g_result,res_imgfile):
    fig = plt.figure(figsize=(15,10))
    plt.plot(df_rlPred['g_ptsx']*1000,df_rlPred['g_ptsy'], color = 'green',label="RL Prediction")
    plt.plot(df_nom['g_ptsx']*1000,df_nom['g_ptsy'], color = 'blue', label="Nominal")
    plt.plot(x_i*1000,g_result,linestyle="dashed", dashes=(5, 5),color = 'orange', label="GA Optimized")

    plt.legend(loc='upper right',fontsize=20)
    plt.ylabel('g points V(x,y=c) [Volt]',fontsize=20)
    plt.xlabel('x axis [mm]',fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.savefig(f"{res_imgfile}")
    plt.close()

# function to plot evaluation stats
#--------------------------------------------------------------------------------------    
def plot_evaluation_stats(steps,res,eval_imgfile):
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history]).flatten()
    #opt_F = np.array([e.pop.get("F") for e in res.history])
    opt_avg = ([np.mean(e) for e in opt])
    
    fig = plt.figure(figsize=(15,10))
    plt.plot(steps, opt, color = 'blue',label="Convergence")
    #plt.plot(np.linspace(0,1000,1001), opt_avg, ":", label="Average Energy (Fitness)")
    #plt.yscale("log")
    plt.legend(loc='upper right',fontsize=20)
    plt.ylabel('energy [VAs]',fontsize=20)
    plt.xlabel('optimization step',fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.savefig(f"{eval_imgfile}")
    plt.close()

# function to generate initial sample
#--------------------------------------------------------------------------------------    
def custom_initializer(n_samples, noise_scaling_factor,g):
    # Your custom initialization strategy goes here
    # For example, random initialization in the range [0, 1]
    g_new = g[2:-1]+np.random.uniform(low=0.0, high=1.0, size=(n_samples,1))*noise_scaling_factor*g[2:-1]
    g_new[g_new>1] = 1
    return g_new

# function to callback for best solutions
#--------------------------------------------------------------------------------------
def my_callback(algorithm):
    best_solution = algorithm.pop[np.argmin(algorithm.pop.get("F")[:, 0])]  # Get the best solution
    best_solutions.append(best_solution.X)  # Store the best solution for the current generation

# main function
#--------------------------------------------------------------------------------------
def optimizer(a,b,c,d,N,V0,eps_r1,eps_r2,df_rlPred,population_size,evolution_steps):
    
    # extarct x and y coordinates of initial curve
    #---------------------------------------------
    x_i = np.array(df_rlPred['g_ptsx']) #x-coordinates
    g = np.array(df_rlPred['g_ptsy']) #y-coordinates

    # set up the lower and upper threshold
    #--------------------------------------------
    lower = list(np.zeros((len(x_i)-3,),np.int64))
    upper = GAopt.line_equ(d,a,np.array(x_i[2:-1]))
    
    # set the thrshold coordinates
    #---------------------------------------------
    x_threshold = np.array([0,d]+list(x_i[2:-1])+[a])
    y_threshold = np.array([1,1]+list(upper)+[0])
    
    
    # initialize the GA Problem class
    #----------------------------------------------
    problem = GAopt.GA_Optimizer(a,b,c,d,N,V0,x_i,eps_r1,eps_r2,lower,upper)
    
    # Choose Optimization Algorithm and Termination Criterion
    # population_size = 100
    # evolution_steps = 1000

    noise_scaling_factor = 0
    n_samples=population_size
    algorithm = GA(pop_size=population_size,
                max_gen=1000000,
                sampling=custom_initializer(n_samples,noise_scaling_factor,g))
    termination = get_termination("n_eval", evolution_steps*population_size)
    
    # optimization run
    #--------------------------------------------------
    start_time = time.time()

    res = minimize(problem,
                algorithm,
                termination,
                callback=my_callback,
                seed=1,
                verbose=True,
                save_history=True)

    runtime = round((time.time()-start_time),2)
    print("Total runtime: "+str(runtime)+" s")

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    
    # OUTPUT
    #-------------------------------------------------------
    # concatenate with fixed point values
    g_result = np.concatenate((np.ones(2),res.X,np.zeros(1)))
    
    # make Result folder
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    
    # image file paths
    imgfile = results_folder+f'/GA_Result_GPts_{case}.png'
    eval_imgfile = results_folder+f'/GA_Result_Eval_{case}.png'
        
    # plot the results
    plot_result(x_i,df_rlPred,df_nom,g_result,imgfile)
    steps = np.linspace(1,evolution_steps,evolution_steps+1)
    plot_evaluation_stats(steps,res,eval_imgfile)
    
    result = {
        'g_ptsx': x_i,
        'g_ptsy': g_result
        }
    df_result = pd.DataFrame(result)
    df_result['energy'] = res.F[0]
    df_result['time'] = runtime
    
    resultfile = results_folder+f'/GA_Result_GPts_{case}.csv'
    df_result.to_csv(f'{resultfile}')
    
    return df_result
    
    