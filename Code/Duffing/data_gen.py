# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:03:54 2021

@author: wilke
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
#import multiprocessing as mp

import progressbar as pb

"""
Duffing Oscillator Equation of motion, given x0,v0 and t it returns x_dot, v_dot
"""
def eom(u, t, gamma):
    alpha=-1
    beta=1
    delta=0.3
    gamma=gamma
    omega=1.2
    x, dx = u[0], u[1]
    ddx= gamma * np.cos(omega * t) - (delta * dx + alpha*x + beta * x**3)
    return [dx,ddx]


"""
Samples randomly from x0 in [-2,2], v0 in [-1,1]
"""

def sample_many_traj(num_samples, gamma = 0.37):
    #Initialise the output arrays
    X = np.empty((num_samples, 3))
    y = np.empty((num_samples, 2))
    #Define bounds of the sampling
    x_max = 2
    x_min = -2
    v_max = 1
    v_min = -1
    #Define the t_range to draw from
    t_range = np.linspace(0, 50, 500, endpoint=False)
    #Generate num_samples samples
    with pb.ProgressBar(max_value=num_samples) as bar:
        for i in range(num_samples):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min
            #Generate a trajectory
            func = lambda u,t: eom(u, t, gamma)
            trajectory = odeint(func, [x0,v0], t_range)
            #Sample a random point along the trajectory
            t2_index = np.random.randint(0, len(t_range))
            X[i,:] = [x0, v0, t_range[t2_index]]
            y[i,:] = trajectory[t2_index,:]
            bar.update(i)
            
    return X, y


def main():
    #Generate the data
    X_train, y_train = sample_many_traj(int(1e6))
    X_test, y_test = sample_many_traj(int(1e5))
    
    #Save the generated data in pd dataframes
    pd.DataFrame(X_train, columns=['x0','v0','t']).to_csv("X_train_euler.csv")
    pd.DataFrame(y_train, columns=['xt','vt']).to_csv("y_train_euler.csv")
    pd.DataFrame(X_test, columns=['x0','v0','t']).to_csv("X_test_euler.csv")
    pd.DataFrame(y_test, columns=['xt','vt']).to_csv("y_test_euler.csv")


if __name__ == "__main__":
    main()