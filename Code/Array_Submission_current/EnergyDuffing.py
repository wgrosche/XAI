# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft


import shap as shap
try:
    import lime
    import lime.lime_tabular    
except ImportError:
    pass

# Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Sequence
from tensorflow import keras

# for reproducibility of this notebook:
rng = np.random.RandomState(42)
tf.random.set_seed(42)
np.random.seed(42)


import os
import sys



class Duffing():
    """
        Class for the Duffing Oscillator
    """
    def __init__(self, parameters = {'alpha': [0.3], 'beta': [-0.1], 'gamma': [0.37], 'delta': [0.3], 'omega': [1.2]}, 
                 labels = ['xt','vt'], features = ['x0','v0', 't', 'energy'], scaler = None):
        """
            Define Parameter Configuration to Model

            Parameters
            ----------
            alpha : float, linear stiffness
            beta  : float, non linearity in the restoring force
            gamma : float, amplitude of the periodic driving force
            delta : float, amount of damping
            omega : float, angular frequency of the periodic driving force
        """   
        self.labels = labels
        self.features = features
        self.scaler = scaler
        self.parameters = parameters
        self.suffix = "_"+str(parameters['alpha'])+"_"+str(parameters['beta'])+"_"+str(parameters['gamma'])+"_"+str(parameters['delta'])+"_"+str(parameters['omega'])


            
        
    def eom(self, t, u):
        """
            Duffing Oscillator Equation of Motion

            ddx + delta * dx + alpha * x + beta * x**3 = gamma * cos(omega * t)

            Input
            ----------
            u : vector of length 2, (x,v)
                Position and Velocity at time t
            t : float, the time t

            Returns
            ----------
            [dx,ddx] : Tuple, Time derivatives of 
                        position and velocity at time t
        """
        x, dx = u[0], u[1]
        ddx = (self.parameters['gamma'] * np.cos(self.parameters['omega'] * t) - (self.parameters['delta'] * dx + self.parameters['alpha'] * x + self.parameters['beta'] * x**3))

        return [dx,ddx]
    
    def energy(self, x, v):
        return 0.5*v**2 + 0.5*self.parameters['alpha']*x**2 +0.25*self.parameters['beta']*x**4

    def termination_event(self, t, y):
        """
            Stops Numerical Integration once points wander too far away
        """
        return (np.abs(y[0]) - 30)*(np.abs(y[1]) - 30)
    termination_event.terminal = True


    def generate(self, num_samples = int(5e1), samples=10, end_time=100, gridded=False, num_gammas = 1):
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
        self.scaler = None
        #Define bounds of the sampling
        x_min = -2
        x_max = 2
        v_min = -2
        v_max = 2
        #Initialise the output arrays        
        X = np.empty((num_samples*samples, len(np.hstack((self.features, self.labels)))))
        #Define the t_range to draw from
        t_range = np.linspace(0, end_time, 100, endpoint=False)
        t_vals = np.sort(np.random.choice(t_range, size = samples, replace=False))

        #Generate num_samples samples
        for i in tqdm(range(num_samples), desc="Generating Data…", ascii=False, ncols=75):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min

            #Generate a trajectory
            trajectory = solve_ivp(self.eom, [0, end_time], [x0,v0], t_eval = t_vals, events = [self.termination_event])
            traj_cutoff =  samples - len(trajectory.y[0])
            traj_x = np.append(trajectory.y[0].reshape(-1,1), trajectory.y[0][-1]*np.ones(traj_cutoff).reshape(-1,1))
            traj_v = np.append(trajectory.y[1].reshape(-1,1), trajectory.y[1][-1]*np.ones(traj_cutoff).reshape(-1,1))
            val_range_low = i*samples
            val_range_high = (i+1)*samples
            X[val_range_low:val_range_high,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                           v0*np.ones(samples).reshape(-1,1),
                                           t_vals.reshape(-1,1),
                                           self.energy(x0, v0)*np.ones(samples).reshape(-1,1),
                                           traj_x.reshape(-1,1), 
                                           traj_v.reshape(-1,1)))
        
        self.X_df = pd.DataFrame(X, columns = np.hstack((self.features, self.labels)))
        return self.X_df

    def scale_features(self):
        if self.scaler == None:
            self.scaler = MinMaxScaler(feature_range=[-1,1])
            self.X_df[self.features] = self.scaler.fit_transform(self.X_df[self.features].values)
        else: return



    def predict(self, X):
        if self.scaler == None:
            self.scale_features()
        if type(X) == pd.core.frame.DataFrame:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X.values), columns=X.columns)
        elif type(X) == pd.core.series.Series:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X.values.reshape(1,-1)), columns=X.index)
        elif type(X) == np.ndarray:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.features)

        y = np.ones((np.shape(X_temp)[0], 2))
        for i in range(0,np.shape(X_temp)[0]):
            traj = solve_ivp(self.eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], 
                            t_eval = None, events = [self.termination_event])
            y[i] = [traj.y[0][-1], traj.y[1][-1]]
            
        return y

    def predict_x(self, X):
        return self.predict(X)[:,0]

    def predict_v(self, X):          
        return self.predict(X)[:,1]

        
    def vals_to_df(self, values, data, explainer = "lime", suffix = None):
        df = pd.DataFrame(values[0], columns = [self.labels[0] + "_" + i for i in self.features])
        df = df.join(pd.DataFrame(values[1], columns = [self.labels[1] + "_" + i for i in self.features]))
        df = df.join(pd.DataFrame(data.values, columns = self.features))
        df.insert(df.shape[1], 'explainer' ,[explainer for _ in range(df.shape[0])])
        return df