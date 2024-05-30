# imports
#----------------------------
import streamlit as st
from streamlit.components.v1 import html
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# import RL
#-----------------------------------------------
from stable_baselines3 import SAC
from MicroStripArrangementEnv import MicroStripArrangementEnv as msaEnv

# import GA
#-----------------------------------------------
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

import gentic_algorithm_optimizer as GAopt

# import interpolation
#-----------------------------------------------
from scipy.interpolate import interp1d

# set up the app title
#--------------------------------
st.title('Microstrip Potential Curve Shape Optimisation')
st.markdown('''
This program runs the application to predict the near-global solution with RL (SAC) and further optimise with GA

''')


###############################################################
# GENERAL
###############################################################
# load model
#-----------------------------------------
SAC_model = SAC.load("model\SAC_MSAEnv.zip")

# app columns
#----------------------------
col1, col2 = st.columns(2)

# environment parameters
#-----------------------------------------
with col1:
    a = st.number_input("half width arrangement (a)",value=1.38e-3)
    st.write("half width arrangement (a)", a)

    c = st.number_input("height of substrate (c)",value=0.1382e-3)
    st.write("height of substrate (c)", c)

    er_1 = st.number_input("dielctric constant air (er_1)",value=1.0)
    st.write("dielctric constant air (er_1)", er_1)

    num_fs = st.number_input("number of fourier coefficients",value=2000)
    st.write("number of fourier coefficients", num_fs)
    
    population_size = st.number_input("population size",value=100)
    st.write("population size", population_size)

with col2:    
    b = st.number_input("height of arrangement (b)",value=2.76e-3)
    st.write("height of arrangement (b)", b)
    
    d = st.number_input("half width microstrip (d)",value=0.05e-3)
    st.write("half width microstrip (d)", d)
    
    er_2 = st.number_input("dielctric constant medium (er_2)",value=12.9)
    st.write("dielctric constant medium (er_2)", er_2)
    
    num_pts = st.number_input("number of G points",value=53)
    st.write("number of G points", num_pts)
    
    evolution_steps = st.number_input("evolution steps",value=1000)
    st.write("evolution steps", evolution_steps)
    
###############################################################
# RL
###############################################################
def rl_predict(model,env):
    # env reset
    ######################################################
    obs_space,_info = env.reset()
    
    # predict
    #######################################################
    action, _states = model.predict(obs_space)
    action = np.abs(action)
    
    # get the g points from RL agent
    #######################################################
    g_ptsx,g_ptsy,control_points = env.g_points_cubic_bezier(action)

    # add the zeroth position values ((0,1) to g_ptsx and g_ptsy)
    ##############################################################
    g_ptsx = np.insert(g_ptsx,0,0)
    g_ptsy = np.insert(g_ptsy,0,1)
    
    # convert to dataframe
    ################################################
    SAC_res = {'g_ptsx':g_ptsx,
           'g_ptsy':g_ptsy
           }
    
    return pd.DataFrame(SAC_res)

###############################################################
# GA
###############################################################
# initialize the best solution for GA
#----------------------------------------------
best_solutions = []
    
# function to generate initial sample
#--------------------------------------------------------------------------------------    
def custom_initializer(n_samples, noise_scaling_factor,g):
    # Your custom initialization strategy goes here
    # For example, random initialization in the range [0, 1]
    g_new = g[2:-1]+np.random.uniform(low=0.0, high=1.0, size=(n_samples,1))*noise_scaling_factor*g[2:-1]
    g_new[g_new>1] = 1
    return g_new

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
        self.best_fmin = []
        self.plot_placeholder = st.empty()  # Placeholder for the plot

    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)
        self.best_fmin.append(algorithm.pop.get("F").min())
        self.plot_convergence()
        
    # function to plot convergence
    #-------------------------------------------------------------------------------------- 
    def plot_convergence(self):
        fig = px.line(x=np.arange(len(self.best_fmin)), y=self.best_fmin, markers=True)
        fig.update_layout(
        title='Convergence',
        xaxis_title='steps',
        yaxis_title='energy [VAs]')
        
        self.plot_placeholder.plotly_chart(fig, use_container_width=True)

# GA problem initializer
#--------------------------------------------------------------------------------------
def ga_problem_init(a,b,c,d,N,V0,eps_r1,eps_r2,df_rlPred):
    
    # extarct x coordinates of initial curve
    #---------------------------------------------
    x_i = np.array(df_rlPred['g_ptsx']) #x-coordinates

    # set up the lower and upper threshold
    #--------------------------------------------
    lower = list(np.zeros((len(x_i)-3,),np.int64))
    upper = GAopt.line_equ(d,a,np.array(x_i[2:-1]))
    
    
    # initialize the GA Problem class
    #----------------------------------------------
    problem = GAopt.GA_Optimizer(a,b,c,d,N,V0,x_i,eps_r1,eps_r2,lower,upper)
    
    return problem
def ga_run(df_rlPred):
    # initialize the GA problem
    #############################################
    problem = ga_problem_init(a=a,b=b,c=c,d=d,N=num_fs,V0=1,eps_r1=er_1,eps_r2=er_2,df_rlPred=df_rlPred)
    
    # extarct y coordinates of initial curve
    #########################################
    g = np.array(df_rlPred['g_ptsy']) #y-coordinates
    
    # setup GA optimisation algorithm with GA problem and initial curve
    ###################################################################
    noise_scaling_factor = 0
    n_samples=population_size
    algorithm = GA(pop_size=population_size,
                max_gen=1000000,
                sampling=custom_initializer(n_samples,noise_scaling_factor,g))
    termination = get_termination("n_eval", evolution_steps*population_size)

    # Initialize the callback
    my_callback = MyCallback()
    
    res = minimize(problem,
                algorithm,
                termination,
                callback=my_callback,
                seed=1,
                verbose=True,
                save_history=True)
    
    return res

###############################################################
# Interpolate
###############################################################

def interpolate(df_rlPred,g_result):
    # get the interpolation
    bezier_interpolated = interp1d(df_rlPred['g_ptsx'][1:].values, g_result[1:])    
    
    # generate x coordinates in exponential distribution
    t_temp = np.linspace(0,1,int(num_pts))
    g_ptsx_interp = d + (a - d)*np.power(t_temp,3)
    
    g_ptsy_interp = bezier_interpolated(g_ptsx_interp)
    
    g_ptsx_interp = np.insert(g_ptsx_interp,0,0)
    g_ptsy_interp = np.insert(g_ptsy_interp,0,1)
    
    df_result = pd.DataFrame({
        'g_ptsx':g_ptsx_interp,
        'g_ptsy':g_ptsy_interp
    })
    
    V0 = 1
    v_n = GAopt.calculateSmallV(a,b,c,d,num_fs,V0,g_ptsx_interp,g_ptsy_interp)
    energy = GAopt.calculateW(v_n,a,b,c,num_fs,er_1,er_2)
    return df_result, energy

###############################################################
# MAIN
###############################################################
def main(a,b,c,d,er_1,er_2,num_fs,model):
    
    if st.button('Process'):
        # initialize environment object
        #############################################
        env = msaEnv(V0=1,hw_arra=a,ht_arra=b,ht_subs=c,hw_micrstr=d,ht_micrstr=0,er_1=er_1,er_2=er_2,num_fs=num_fs,num_pts=53)
        
        # Predict initial curve
        ##############################################
        df_rlPred = rl_predict(model,env)
        
        # GA optimise
        ##############################################
        res = ga_run(df_rlPred)
        
        # concatenate with fixed point values
        g_result = np.concatenate((np.ones(2),res.X,np.zeros(1)))

        # Interpolate for more points
        ##############################################
        df_result, energy = interpolate(df_rlPred,g_result)
        
        
        with col1:
            st.subheader("Minimum Energy Potential Curve G-Points")
            st.dataframe(df_result)
        with col2:
            st.text(f'Minimum Energy Achived {energy} VAs')
            
            # Plotting using plotly
            fig = px.line(df_result, x='g_ptsx', y='g_ptsy', markers=True)
            fig.update_layout(
            title='Minimum Energy Potential Curve',
            xaxis_title='x [mm]',
            yaxis_title='potential [V]')
            # Display the plot in Streamlit
            st.plotly_chart(fig)
        
        
if __name__ == '__main__':
    main(a,b,c,d,er_1,er_2,num_fs,SAC_model)