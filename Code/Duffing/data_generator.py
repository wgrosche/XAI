# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:03:54 2021

@author: wilke

Data Generation for the Duffing Oscillator in an Explainable AI setting.

"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import progressbar as pb


def eom(u, t):
    """
        Duffing Oscillator Equation of Motion

        ddx + delta * dx**2 + alpha * x + beta * x**3 = gamma * cos(omega * t)

        Input
        ----------
        u : vector of length 2, (x,v)
            Position and Velocity at time t
        t : float, the time t

        Parameters
        ----------
        alpha : float, linear stiffness
        beta  : float, non linearity in the restoring force
        gamma : float, amplitude of the periodic driving force
        delta : float, amount of damping
        omega : float, angular frequency of the periodic driving force

        Returns
        ----------
        [dx,ddx] : Tuple, Time derivatives of 
                    position and velocity at time t

    """
    #Set system parameters
    alpha=-1
    beta=1
    delta=0.3
    gamma=0.37
    omega=1.2
    x, dx = u[0], u[1]
    ddx= gamma * np.cos(omega * t) - (delta * dx + alpha*x + beta * x**3)
    
    return [dx,ddx]




def sample_many_traj(num_samples):
    """
        Generates training samples using scipy.integrate.odeint
        to calculate the temporal evolution of a Duffing system.

        Samples randomly from x0 in [-2,2], v0 in [-1,1].

        For each set of initial conditions we generate a trajectory.
        The trajectory is randomly sampled to generate training
        pairs: X = (x0,v0,t), y = (xt,vt)

        Input
        ----------
        num_samples : int, number of training
                        samples to be generated

        Returns
        ----------
        X : array((num_samples,3)), each entry in the array
            is a training sample (x0,v0,t)
        y : array((num_samples,2)), each entry in the array
            is a target sample (xt,vt)
    """
    #Initialise the output arrays
    delay=0
    X = np.empty((num_samples*(500-delay), 4))
    y = np.empty((num_samples*(500-delay), 2))
    #Define bounds of the sampling
    x_max = 2
    x_min = -2
    v_max = 1
    v_min = -1
    #Define the t_range to draw from
    t_range = np.linspace(0, 100, 500, endpoint=False)
    #Generate num_samples samples
    with pb.ProgressBar(max_value=num_samples) as bar:
        for i in range(num_samples):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min
            #Generate a trajectory
            trajectory = odeint(eom, [x0,v0], t_range)
            for j in range(0,500-delay):
                X[(500-delay)*i+j,:] = [x0, v0, t_range[j+delay], np.random.random_sample()]
                y[(500-delay)*i+j,:] = trajectory[j+delay,:]
            bar.update(i)
            
    return X, y


def main():
    suffix = "delta03_irrelevant"
    #Generate the data
    X_train, y_train = sample_many_traj(int(1e5))
    #Save the generated data in pd dataframes
    pd.DataFrame(X_train, columns=['x0','v0','t','rand']).to_csv("Data/X_train_"+suffix+".csv")
    pd.DataFrame(y_train, columns=['xt','vt']).to_csv("Data/y_train_"+suffix+".csv")


    X_test, y_test = sample_many_traj(int(1e3))
    #Save the generated data in pd dataframes
    pd.DataFrame(X_test, columns=['x0','v0','t','rand']).to_csv("Data/X_test_"+suffix+".csv")
    pd.DataFrame(y_test, columns=['xt','vt']).to_csv("Data/y_test_"+suffix+".csv")


if __name__ == "__main__":
    main()