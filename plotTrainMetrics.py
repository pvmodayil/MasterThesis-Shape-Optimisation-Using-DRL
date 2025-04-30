#######################################
# Author : Philip Varghese Modayil
# Date   : 23.10.2024
# Topic  : Microstrip Arrangement Training Metrics 
#######################################

#####################################################################################
#                                     Imports
#####################################################################################
import argparse
import pandas as pd
from datetime import datetime
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

#####################################################################################
#                                     Functions
#####################################################################################

def plotLoss(steps_actor, values_actor, steps_critic, values_critic, envCase):
    cwd = os.getcwd()
    imgFile = os.path.join(cwd,"training",envCase,"Loss.png")
    dataFile = os.path.join(cwd,"training",envCase,"Loss.csv")
    data = {
        "steps_actor": steps_actor,
        "actor_values": values_actor,
        "steps_critic": steps_critic,
        "critic_values": values_critic,
    }
    (pd.DataFrame(data)).to_csv(dataFile,index=False)
    
    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))

    # Flatten the axs array to easily access each subplot
    axs = axs.flatten()

    axs[0].plot(steps_critic, values_critic,color='orange',label='_Critic Loss',linewidth=3)
    axs[0].set_xlabel('Episode#',fontsize=25)
    axs[0].set_ylabel('Critic Loss',fontsize=25)
    axs[0].tick_params(axis='x', labelsize=25, pad=10)
    axs[0].tick_params(axis='y', labelsize=25, pad=10)

    axs[1].plot(steps_actor, values_actor,color='orange',label='_Actor Loss',linewidth=3)
    axs[1].set_xlabel('Episode#',fontsize=25)
    axs[1].set_ylabel('Actor Loss',fontsize=25)
    axs[1].tick_params(axis='x', labelsize=25, pad=10)
    axs[1].tick_params(axis='y', labelsize=25, pad=10)

    plt.tight_layout()
    
    plt.savefig(imgFile)
    plt.close()

def plotRewardEntropy(steps, values,relative_time,relative_time_name,metric,envCase):
    cwd = os.getcwd()
    imgFile = os.path.join(cwd,"training",envCase,f"{metric}.png")
    dataFile = os.path.join(cwd,"training",envCase,f"{metric}.csv")
    data = {
        "relative_time[Min]": relative_time,
        "steps": steps,
        f"{metric}": values
    }
    (pd.DataFrame(data)).to_csv(dataFile,index=False)
    
    # Plot the reward with twin x-axis
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # Plot against steps_rew on ax1
    ax1.plot(steps, values, color='orange',label='_noLabel',linewidth=3)

    # Plot against relative_hours on ax2 (invisible for demonstration)
    ax2.plot(relative_time, values, alpha=0)

    # Customize plot
    ax1.set_xlabel('Episode#', fontsize=25)
    ax1.set_ylabel(metric, fontsize=25)
    ax2.set_xlabel(relative_time_name, fontsize=25)

    # Set ticks and labels
    ax1.tick_params(axis='x', labelsize=25, pad=10)
    ax1.tick_params(axis='y', labelsize=25, pad=10)
    ax2.tick_params(axis='x', labelsize=25, pad=10)

    # Display grid
    ax1.grid(True)

    # Display legend
    # ax1.legend(fontsize=20)
    # Show plot
    plt.savefig(imgFile)
    plt.close()

def main(envCase: str, logPath: str) -> None:
    # Create an EventAccumulator
    event_acc = EventAccumulator(logPath)
    event_acc.Reload()
    
    # REWARDS
    #######################################################################
    data = event_acc.Scalars('rollout/ep_rew_mean')
    # Extract the steps and values
    wall_time = [event.wall_time for event in data]
    steps_rew = [event.step for event in data]
    values_rew = [event.value for event in data]
    # Convert Unix timestamps to datetime objects
    timestamps_dt = [datetime.fromtimestamp(ts) for ts in wall_time]
    # Calculate relative hours from the first timestamp
    start_time = timestamps_dt[0]
    relative_hours = [(ts - start_time).total_seconds() / 60 for ts in timestamps_dt]
    plotRewardEntropy(steps_rew, values_rew,relative_hours,'Training Time [min]','Reward',envCase)
    
    # ENTROPY COEFFICIENTS
    ########################################################################
    data = event_acc.Scalars('train/ent_coef')
    steps_entcoeff = [event.step for event in data]
    values_entcoeff = [event.value for event in data]
    plotRewardEntropy(steps_entcoeff, values_entcoeff,relative_hours,'Training Time [min]','Entropy Coefficient',envCase)
    
    # LOSS
    ########################################################################
    data = event_acc.Scalars('train/critic_loss')
    steps_critic = [event.step for event in data]
    values_critic = [event.value for event in data]
    
    data = event_acc.Scalars('train/actor_loss')
    steps_actor = [event.step for event in data]
    values_actor = [event.value for event in data]
    
    plotLoss(steps_actor, values_actor, steps_critic, values_critic, envCase)
    
if __name__ == '__main__':
    # python plotTrainMetrics.py -case 'CaseD/L' -path 'FullPathToLogFile'
    parser = argparse.ArgumentParser(description="MicroStripArrangement training metrics")
    parser.add_argument('-case', '--CASE', type=str,  dest='case',default='CaseD', help='CaseD/L')
    parser.add_argument('-path', '--PATH', type=str,  dest='path', help='path to log file')
    args = parser.parse_args()
    
    envCase = args.case
    logPath = args.path
    
    main(envCase,logPath)