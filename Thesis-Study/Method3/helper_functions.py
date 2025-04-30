#######################################
# Author : Philip Varghese Modayil
# Date   : 23.01.2024
# Topic  : Helper Functions
#######################################

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from stable_baselines3.common.results_plotter import load_results, ts2xy
from typing import *

# function to plot the results of bezier curve
###################################################################################################################################################
def plot_cubic_bezier(imgdir:str, calc_values:dict, nom_values:dict, control_points:np.ndarray, case:str, time:float) -> None:
    """
        Function plot the prediction and nominal curves
        Input
        -----------
        imgdir: storing location
            str
        calc_values: prediction result dictionary
            dict
        nom_values: nominal result dictionary
            dict
        control_points: cubic bezier_control points
            numpy array
        case: microstrip setup
            str
        time: training time
            float
    
        Output
        -----------
        plots
        """
    # Plot the gpts
    fig_gpts = plt.figure(figsize=(15,10))
    plt.plot(calc_values['X_g_pts']*1000,calc_values['Y_g_pts'], color='green', label='Prediction') 
    plt.plot(nom_values['X_g_pts']*1000,nom_values['Y_g_pts'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm
    plt.scatter(control_points[:,0]*1000,control_points[:,1],color='red',label='Control Points')
    plt.legend(loc='upper right')
    plt.ylabel('g points V(x,y=c) [Volt]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"g points {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']} VAs, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_GPtsPredictionResult.png")
    plt.close()
    
    # Plot the potential
    fig_pot = plt.figure(figsize=(15,10))
    plt.plot(calc_values['x_vals']*1000,calc_values['potential'], color='green', label='Prediction')  
    plt.plot(nom_values['x_vals']*1000,nom_values['potential'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm
    plt.legend(loc='upper right')
    plt.ylabel('potential V(x,y=c) [Volt]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"Potential Profile {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']} VAs, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_PotentialPredictionResult.png")
    plt.close()
    
    # Plot the potential
    fig_sigma = plt.figure(figsize=(15,10))
    plt.plot(calc_values['x_vals']*1000,calc_values['sigma'], color='green', label='Prediction')  
    plt.plot(nom_values['x_vals']*1000,nom_values['sigma'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm
    plt.legend(loc='upper right')
    plt.ylabel('sigma rho(x,y=c) [C/m]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"Charge Density {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']}, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_SigmaPredictionResult.png")
    plt.close()
    
# function to plot the results of decaying exponential curve
###################################################################################################################################################
def plot_decay_exp(imgdir:str, calc_values:dict, nom_values:dict, case:str, time:float) -> None:
    """
        Function plot the prediction and nominal curves
        Input
        -----------
        imgdir: storing location
            str
        calc_values: prediction result dictionary
            dict
        nom_values: nominal result dictionary
            dict
        case: microstrip setup
            str
        time: training time
            float
    
        Output
        -----------
        plots
        """

    # Plot the gpts
    fig_gpts = plt.figure(figsize=(15,10))
    plt.plot(calc_values['X_g_pts']*1000,calc_values['Y_g_pts'], color='green', label='Prediction') 
    plt.plot(nom_values['X_g_pts']*1000,nom_values['Y_g_pts'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm 
    plt.legend(loc='upper right')
    plt.ylabel('g points V(x,y=c) [Volt]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"g points {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']} VAs, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_GPtsPredictionResult.png")
    plt.close()
    
    # Plot the potential
    fig_pot = plt.figure(figsize=(15,10))
    plt.plot(calc_values['x_vals']*1000,calc_values['potential'], color='green', label='Prediction')  
    plt.plot(nom_values['x_vals']*1000,nom_values['potential'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm
    plt.legend(loc='upper right')
    plt.ylabel('potential V(x,y=c) [Volt]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"Potential Profile {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']} VAs, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_PotentialPredictionResult.png")
    plt.close()
    
    # Plot the potential
    fig_sigma = plt.figure(figsize=(15,10))
    plt.plot(calc_values['x_vals']*1000,calc_values['sigma'], color='green', label='Prediction')  
    plt.plot(nom_values['x_vals']*1000,nom_values['sigma'], color='blue', label='Nominal')  # multiplied with 1000 to convert to mm
    plt.legend(loc='upper right')
    plt.ylabel('sigma rho(x,y=c) [C/m]')
    plt.xlabel('x axis [mm]')
    plt.grid(True)
    plt.title(f"Charge Density {case}: Nominal Energy:{nom_values['energy']}, Prediction Energy:{calc_values['energy']}, Learning Time:{time}")
    plt.savefig(f"{imgdir}/{case}_SigmaPredictionResult.png")
    plt.close()

# characteristics calculation
###################################################################################################################################################
def characteristics_calc(env,g_ptsx:np.ndarray, g_ptsy:np.ndarray, x_plot:np.ndarray) -> Dict[str,Any]:     
    """
        Function to calculate the characteristics of the curve
        Input
        -----------
        env: microstrip setup environment
            class object
        g_ptsx: x-axis values of the curve
            numpy array
        g_ptsy: y-axis values of the curve
            numpy array
        x_plot: x-axis values for plotting
            numpy array
    
        Output
        -----------
        dicionary of results
        """
    # characterisitc calculation
    ##############################################################
    # potentila coefficients
    vn = env.potential_coeff_parallel(g_ptsy[2:-1],g_ptsx[2:-1]) # compensated for those values within the formula
    # calculate energy
    energy = env.energy(vn)
    # potential profile
    potential_profile = env.potential(vn,x_plot)
    # charge density
    sigma_charge_density = env.charge_density_parallel(vn,x_plot)
    sigma_charge_density = np.array([float(sigma) for sigma in sigma_charge_density]) # convert mpf object to float

    mew_capacitance = env.capacitance(energy)
    
    # result dictionary
    ##############################################################
    result_values = {
        'x_vals':x_plot,
        'X_g_pts':g_ptsx,
        'Y_g_pts':g_ptsy,
        'energy':energy,
        'potential':potential_profile,
        'sigma':sigma_charge_density,
        'capacitance':mew_capacitance
    }
    
    return result_values

# save data
###############################################################################################################################################################
def save_data(result_dic:dict, case:str, algo:str) -> None:
    """
        Function to save data
        Input
        -----------
        result_dic: result
            dict
        case: microstrip environment setup
            str
        
        Output
        -----------
        None
        """
    try:
        op_loc = f'training/{algo}_{case}_Result_Training.csv'
        df_result = pd.DataFrame([result_dic])

        if os.path.isfile(op_loc):
            df_prev= pd.read_csv(op_loc)
            df_result = pd.concat([df_prev, df_result], ignore_index=True)
        
        df_result.index = [0]
        df_result.to_csv(op_loc,index=False)
        print('data saved........')
    except:
        print('data not saved error.......')
    
# predict and calculate
##################################################################################################################################################################
def do_prediction(model:Any, env:Any, x_true:np.ndarray, y_true:np.ndarray, imgdir:str, case:str, time_train:float) -> Tuple[dict,dict,np.ndarray]:
    """
        Function to predict and calculate the characteristics of the curve
        Input
        -----------
        model: RL model
            object
        env: microstrip setup environment
            class object
        x_true: x-axis values of the nominal curve
            numpy array
        y_true: y-axis values of the nominal curve
            numpy array
        imgdir: location to store images
            str
        case: microstrip environment setup
            str
        time_train: training time
            float
    
        Output
        -----------
        results
        """
    # env reset
    ######################################################
    obs_space,_ = env.reset()
    env.num_pts = 500
    env.num_fs = 1000
    
    # predict
    #######################################################
    action, _states = model.predict(obs_space)
    # g_pts generation
    g_ptsx,g_ptsy,control_points = env.g_points_cubic_bezier(action)
    
    # x values for plot concentrated near end of microstrip
    #############################################################
    num_pts = 100
    t_temp = np.linspace(0,1,int(num_pts/2))
    t_temp = np.linspace(0,1,int(num_pts/2))
    x_plot =   np.array(list(-1*(0 + (env.hw_micrstr)*np.power(t_temp,3)) + env.hw_micrstr)[::-1])
    #self.x = np.linspace(0,0.9*self.hw_micrstr,int(num_pts/2))
    x_plot = np.append(x_plot,env.hw_micrstr + (env.hw_arra - env.hw_micrstr)*np.power(t_temp,3))
    
    # characterisitc calculations
    #######################################################
    # add the zeroth position values ((0,1) to g_ptsx and g_ptsy)
    g_ptsx = np.insert(g_ptsx,0,0)
    g_ptsy = np.insert(g_ptsy,0,1)
    
    calc_values = characteristics_calc(env,g_ptsx,g_ptsy,x_plot)
    nom_values = characteristics_calc(env,x_true,y_true,x_plot)
    
    # send for plotting(make sure the x values and y values are in metre scale and not in mm)
    ########################################################
    plot_cubic_bezier(imgdir,calc_values,nom_values,control_points,case,time_train)

    return calc_values,nom_values,action

# train
#########################################################################################################################################################################
def train_model(algo:str, model:Any, env:Any, case:str, TIMESTEPS:float, x_true:np.ndarray, y_true:np.ndarray, models_dir:str, imgdir:str) -> Tuple[Any,dict,np.ndarray]:
    """
        Function to train model
        Input
        -----------
        algo: ALgorithm used
            str
        model: model created
            Any
        env: microstrip environment
            Any
        case: microstrip environment setup
            str
        TIMESTEPS: training timesteps
            float
        x_true: x-axis values of the nominal curve
            np.ndarray
        y_true: y-axis values of the nominal curve
            np.ndarray
        models_dir: location to ssave model
            str
        imgdir: location to save images
            str
        
        Output
        -----------
        Results
        """
    print('training...........')
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS,log_interval=4,reset_num_timesteps=True, tb_log_name=case,progress_bar=True)
    end_time = time.time()

    model.save(f"{models_dir}/{case}_{algo}_MSAEnv")

    # do prediction
    print('predicting..........')
    calc_values,nom_values,action = do_prediction(model,env,x_true,y_true,imgdir,case,end_time-start_time) 

    print('saving data...........')
    # create dictionary to store values
    result_dic = {'MSAenv':case,
                  'Algorithm':algo, 
                  'Time_Completion':end_time-start_time, 
                  'Energy':calc_values['energy'], 
                  'Energy_True':nom_values['energy'], 
                  'Energy_Diff':calc_values['energy'] - nom_values['energy'],
                  'Capacitance':calc_values['capacitance'],
                  'Capacitance_True':nom_values['capacitance']}

    # save the data
    save_data(result_dic,case,algo)
    
    return model,result_dic,action

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(logdir, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(logdir), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()