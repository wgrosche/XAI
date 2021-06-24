# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:03:54 2021

@author: wilke

Data Generation for the Duffing Oscillator in an Explainable AI setting.

"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
from tqdm import tqdm

class DataGenerator():
    def __init__(self, features = ['x0','v0','t','rand'], 
                 labels = ['xt','vt'], 
                 x_range=[-2,2], v_range = [-1,1]):
        self.features = features
        self.labels = labels
        self.x_range = x_range
        self.v_range = v_range
    
    def generate(self, num_samples = int(1e3), delay=0, samples=500, end_time=100, 
                 params = {'alpha' : [-1],'beta' : [1], 'gamma' :[0.37], 'delta' : [1], 'omega' : [1.2]}):
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
        self.params = params
        self.X = np.empty((num_samples*(samples-delay), (len(self.features)+len(params))))
        self.y = np.empty((num_samples*(samples-delay), len(self.labels)))
        complete_feature_vec = self.features.copy()
        for k in params:
            complete_feature_vec.append(k)
        #Define bounds of the sampling
        x_min = self.x_range[0]
        x_max = self.x_range[1]
        v_min = self.v_range[0]
        v_max = self.v_range[1]
        #Define the t_range to draw from
        t_range = np.linspace(0, end_time, samples, endpoint=False)
        #Generate num_samples samples
        for i in tqdm(range(num_samples), desc="Generating Data…", ascii=False, ncols=75):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min
            for alpha in params['alpha']:
                for beta in params['beta']:
                    for gamma in params['gamma']:
                        for delta in params['delta']:
                            for omega in params['omega']:
                                self.current_params = {'alpha' : alpha,'beta' : beta, 'delta' : delta, 
                                                       'gamma' :gamma, 'omega' : omega}
                                #Generate a trajectory
                                trajectory = odeint(self.eom, [x0,v0], t_range)
                                for j in range(0,samples-delay):
                                    self.X[(samples-delay)*i+j,:] = [x0, v0, t_range[j+delay], 
                                                                     np.random.random_sample(), alpha, beta, gamma,
                                                                     delta, omega]
                                    self.y[(samples-delay)*i+j,:] = trajectory[j+delay,:]
        
        
        self.X = pd.DataFrame(self.X, columns = complete_feature_vec)
        self.y = pd.DataFrame(self.y, columns = self.labels)
        return self.X, self.y


    def save(self, suffix=None, filepath = "Data/"):
        self.X.to_csv(filepath+"X"+suffix+".csv")
        self.y.to_csv(filepath+"y"+suffix+".csv")
        
    def eom(self, u, t):
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
        x, dx = u[0], u[1]
        ddx = (self.current_params['gamma'] * np.cos(self.current_params['omega'] * t) - 
               (self.current_params['delta'] * dx + self.current_params['alpha']*x + self.current_params['beta'] * x**3))
    
        return [dx,ddx]

