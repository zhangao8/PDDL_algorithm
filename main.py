import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import copy
import datetime

from systems import *
from policy_distributions import *


########################
## Training FUNCTIONS ##
########################

def run_sys(sys, policies, num_iter, num_reruns=1, batch_size=64):
    history_list = []
    for i in range(num_reruns):

        history_dict = {}
        for policy_name, _ in policies.items():
            policy_dict = {'lambd': [],
                           'f0': [],
                           'f1': [],
                           'L': []}
            history_dict[policy_name] = policy_dict

        for k in range(num_iter):
            
            states = sys.sample(batch_size)
            
            if k%1000 == 0:
                print("Iteration " + str(k))

            for policy_name, policy in policies.items():            
                
                StartTime = datetime.datetime.now()
                
                actions = policy.get_action(states)
                
                endTime = datetime.datetime.now()
                
                timeElapsed = abs(endTime - StartTime).total_seconds()  
                
                f0 = sys.f0(states, actions)
                f1 = sys.f1(states, actions)
                L = f0 + np.dot(f1, policy.lambd)                
                lambd1 = copy.deepcopy(policy.lambd)

                history_dict[policy_name]['lambd'].append(lambd1.reshape((2)))
                history_dict[policy_name]['f0'].append(-np.mean(f0))
                history_dict[policy_name]['f1'].append(np.mean(f1,0))
                
                history_dict[policy_name]['L'].append(np.mean(L))
                
                policy.learn(states, actions, f0, f1)

        history_list.append(history_dict)
        
    return history_list


def plot_data(data, save_prefix='images/temp_'):
    # plotting variables over time

    #for data_name in ['lambd', 'f0', 'f1', 'L']:
    data_name1 = 'lambd'
    plt.cla()

    data_list = []
    for policy_name, _ in data.items():
        plt.plot(data[policy_name][data_name1], label=policy_name)

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel(data_name1)
    plt.title(data_name1)
    save_file_name = save_prefix + data_name1 + '.png'
    plt.savefig(save_file_name, bbox_inches='tight')
    
    data_name2 = 'f0'
    plt.cla()

    data_list = []
    for policy_name, _ in data.items():
        plt.plot(data[policy_name][data_name2], label=policy_name)

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Objective function value')
    #plt.title('Objective function')
    save_file_name = save_prefix + data_name2 + '.png'
    plt.savefig(save_file_name, bbox_inches='tight')
    
    data_name3 = 'f1'
    plt.cla()

    data_list = []
    for policy_name, _ in data.items():
        plt.plot(data[policy_name][data_name3], label=policy_name)

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Constraint function value')
    #plt.title(data_name)
    save_file_name = save_prefix + data_name3 + '.png'
    plt.savefig(save_file_name, bbox_inches='tight')
    
    data_name4 = 'L'
    plt.cla()

    data_list = []
    for policy_name, _ in data.items():
        plt.plot(data[policy_name][data_name4], label=policy_name)

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel(data_name4)
    plt.title(data_name4)
    save_file_name = save_prefix + data_name4 + '.png'
    plt.savefig(save_file_name, bbox_inches='tight')
    

def save_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['lambd', 'f0', 'f1', 'L']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)



##########################
## Define FSO FUNCTIONS ##
##########################    

def wireless_capacity_powerrelay():
    
    # Define system parameters
    Tf = 1e-8   
    B=5e8
    PT = 1.5
    PS = 0.6    
    e = 1.6e-19    
    A = 1    
    mu = -1 
    sigma = 0.25
    hl=0.1    
    num_channels = 5 
    num_relays = 5
    r=0.75

    # Define FSO system
    sys = FSOCapacitySystem(num_channels, hl, num_relays, PS=PS, PT=PT, e=e, A=A, r=r, Tf=Tf, B=B, mu=mu, sigma=sigma)

    # Define policy distribution
    distribution = CategoryGaussianDistribution(sys.action_dim,  num_relays, num_channels,
        lower_bound=np.ones(int(sys.action_dim/3))*0, 
        upper_bound=np.ones(int(sys.action_dim/3))*PS)
    
    # Define learning policy
    reinforce_policy = LearningPolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim, 
        model_builder=mlp_model,
        distribution=distribution)

    # Primal-Dual Training
    policies = {'Primal-dual deep learning method': reinforce_policy}
    history_dict = run_sys(sys, policies, 20000)[0]
    
    # Save data
    plot_data(history_dict, "images/FSO_capacity_")
    save_data(history_dict, "data/FSO_powerrelay_PDDL.mat")


if __name__ == '__main__':

    import argparse
    import sys

    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(1)

    wireless_capacity_powerrelay()

