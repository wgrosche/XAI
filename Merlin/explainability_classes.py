# import libraries

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


from sklearn.preprocessing import MinMaxScaler


from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft


import matplotlib.pylab as plt
import seaborn as sns
import mpl_interactions.ipyplot as iplt
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)


import shap as shap
try:
    import lime
    import lime.lime_tabular    
except ImportError:
    pass

# Enable Jupyter Notebook's intellisense
%config IPCompleter.greedy=True
%matplotlib inline

%matplotlib notebook
from ipywidgets import *
import matplotlib
import matplotlib.pyplot as plt

# for reproducibility of this notebook:
rng = np.random.RandomState(42)
#tf.random.set_seed(42)
np.random.seed(42)

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets



class NumericExplainer():
    """
        Pretty Brute force numerical gradient calculation for
        explainability of a known function
    """
    def __init__(self, f, features, labels,  h=0.01):
        """
            Initialises with some configurations for the gradient calculation
            as well as the function being differentiated.
            
            Inputs
            --------
            f : function that takes a pandas.DataFrame and outputs a 2d np.array.
            features : list of features in the pd.DataFrame for which we are to 
                differentiate f.
            labels : list of features in the np.array.
        """
        self.f = f
        self.features = features
        self.labels = labels
        self.h = h
        
    def gradient(self, X_val, feature):
        """
            Numerical Gradient Calculation by way of a CFD method.
            Inputs
            --------
            X_val : pandas.DataFrame with columns: features and values at
                which we want to take the numerical gradient.
            feature : feature by which we want to differentiate.
        """
        X_prime_plus = X_val.copy()
        X_prime_plus.loc[:,(feature)] = X_prime_plus[feature] + self.h
        X_prime_minus = X_val.copy()
        X_prime_minus.loc[:,(feature)] = X_prime_minus[feature] - self.h
        
        grad = (self.f(X_prime_plus) - self.f(X_prime_minus))/(2*self.h)
        
        return grad
    def feature_att(self, X):
        """
            Calculates the Gradients for all Entries in X, for each
            feature and label combination.
            
            Inputs
            --------
            X : pandas.DataFrame with columns:features and values at
                which we want to differentiate.
            Returns
            --------
            self.__atts : [np.array[...],np.array[...]] of gradients at
                each of the input points. Calculated for each label and stacked.
        """
        first_run = True
        for i,__label in enumerate(self.labels):
            grads = self.gradient(X, self.features[0])[:,i]
            for __feat in self.features[1:]:
                grads = np.vstack((grads,self.gradient(X, __feat)[:,i]))
            normalised_grads = np.abs(grads)/np.sum(np.abs(grads),axis=0)
            if first_run:
                self.__atts = grads.transpose()
                self.__normalised = normalised_grads.transpose()
                first_run = False
            else:
                self.__atts = [self.__atts, grads.transpose()]
                self.__normalised = [self.__normalised, normalised_grads.transpose()]
                        
        return self.__atts#, self.__normalised
    
    


    
    def event_1(t, y):
    return np.abs(y[0]) - 10

def event_2(t, y):
    return np.abs(y[1]) - 10

event_1.terminal = True
event_2.terminal = True

def generate(num_samples = int(5e1), samples=10, end_time=100, gridded=False):
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
        
        #Define bounds of the sampling
        x_min = -2
        x_max = 2
        v_min = -2
        v_max = 2
        #Initialise the output arrays
        parameter_length = 1
        labels = ['xt','vt']
        features = ['x0','v0','t','rand']
        X = np.empty((num_samples*(samples), len(np.hstack((features, labels)))))
        #X = np.empty((num_samples*(samples), len(features)))
        #y = np.empty((num_samples*(samples), len(labels)))
        #Define the t_range to draw from
        t_range = np.linspace(0, end_time, 100, endpoint=False)
        t_vals = np.sort(np.random.choice(t_range, size = samples, replace=False))
        if gridded:
            x_range = np.linspace(x_min, x_max, int(np.sqrt(num_samples)), endpoint = True)
            v_range = np.linspace(v_min, v_max, int(np.sqrt(num_samples)), endpoint = True)
            #Generate num_samples samples
            for i, x0 in tqdm(enumerate(x_range), desc="Generating Data…", ascii=False, ncols=75):
                for j, v0 in enumerate(v_range):
                    #Generate a trajectory
                    trajectory = solve_ivp(eom, [0, end_time], [x0,v0], t_eval = t_vals, events = [event_1, event_2])
                    traj_cutoff =  samples - len(trajectory.y[0])
                    if traj_cutoff > 0:
                        
                        trajectory.y[0] = np.append(trajectory.y[0].reshape(-1,1), 10.0*np.ones(traj_cutoff))
                        trajectory.y[1] = np.append(trajectory.y[1].reshape(-1,1), 10.0*np.ones(traj_cutoff))
                    X[i*samples:(i+1)*samples,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                                          v0*np.ones(samples).reshape(-1,1), 
                                                          t_vals.reshape(-1,1), 
                                                          np.random.uniform(-1,1,samples).reshape(-1,1),
                                                          trajectory.y[0].reshape(-1,1), 
                                                          trajectory.y[1].reshape(-1,1)))
                    """
                    X[i*samples:(i+1)*samples,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                                          v0*np.ones(samples).reshape(-1,1), 
                                                          t_vals.reshape(-1,1), 
                                                          np.random.uniform(-1,1,samples).reshape(-1,1)))
                    y[i*samples:(i+1)*samples,:] = np.hstack((trajectory.y[0].reshape(-1,1), trajectory.y[1].reshape(-1,1)))
                    """
        else:
            #Generate num_samples samples
            for i in tqdm(range(num_samples), desc="Generating Data…", ascii=False, ncols=75):
                #Generate random starting positions
                x0 = (x_max - x_min) * np.random.random_sample() + x_min
                v0 = (v_max - v_min) * np.random.random_sample() + v_min 
                #Generate a trajectory
                trajectory = solve_ivp(eom, [0, end_time], [x0,v0], t_eval = t_vals, events = [event_1, event_2])
                traj_cutoff =  samples - len(trajectory.y[0])
                if traj_cutoff > 0:
                    x_traj = np.vstack((trajectory.y[0].reshape(-1,1), 10.0*np.ones(traj_cutoff).reshape(-1,1)))
                    v_traj = np.vstack((trajectory.y[1].reshape(-1,1), 10.0*np.ones(traj_cutoff).reshape(-1,1)))
                    X[i*samples:(i+1)*samples,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                                              v0*np.ones(samples).reshape(-1,1), 
                                                              t_vals.reshape(-1,1), 
                                                              np.random.uniform(-1,1,samples).reshape(-1,1),
                                                              x_traj, 
                                                              v_traj))
                else:
                    X[i*samples:(i+1)*samples,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                                              v0*np.ones(samples).reshape(-1,1), 
                                                              t_vals.reshape(-1,1), 
                                                              np.random.uniform(-1,1,samples).reshape(-1,1),
                                                              trajectory.y[0].reshape(-1,1), 
                                                              trajectory.y[1].reshape(-1,1)))
                """
                X[i*samples:(i+1)*samples,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                                          v0*np.ones(samples).reshape(-1,1), 
                                                          t_vals.reshape(-1,1), 
                                                          np.random.uniform(-1,1,samples).reshape(-1,1)))
                #print(trajectory.y[0].reshape(-1,1))
                y[i*samples:(i+1)*samples,:] = np.hstack((trajectory.y[0].reshape(-1,1), trajectory.y[1].reshape(-1,1)))
                """
        
        X_df = pd.DataFrame(X, columns = np.hstack((features, labels)))
        #X_df = pd.DataFrame(X, columns = features)
        #y_df = pd.DataFrame(y, columns = labels)
        #return X_df, y_df, features, labels
        return X_df, features, labels
    
def true_predict(X):
    if type(X) == pd.core.frame.DataFrame:
        X_temp = pd.DataFrame(scaler.inverse_transform(X.values), columns=X.columns)
    elif type(X) == np.ndarray:
        X_temp = pd.DataFrame(scaler.inverse_transform(X), columns=features)
    
    y = np.ones((np.shape(X_temp)[0], 2))
    for i in range(0,np.shape(X_temp)[0]):
        traj = solve_ivp(eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], 
                         t_eval = None, events = [event_1, event_2])
        y[i,:] = [traj.y[0][-1],traj.y[1][-1]]

        
    return y

def true_lime_1(X):
    if type(X) == pd.core.frame.DataFrame:
        X_temp = pd.DataFrame(scaler.inverse_transform(X.values), columns=X.columns)
    elif type(X) == np.ndarray:
        X_temp = pd.DataFrame(scaler.inverse_transform(X), columns=features)
    
    y = np.ones((np.shape(X_temp)[0], 1))
    for i in range(0,np.shape(X_temp)[0]):
        traj = solve_ivp(eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], t_eval = None, events = [event_1, event_2])
        y[i] = traj.y[0][-1]
        
    return y

def true_lime_2(X):
    if type(X) == pd.core.frame.DataFrame:
        X_temp = pd.DataFrame(scaler.inverse_transform(X.values), columns=X.columns)
    elif type(X) == np.ndarray:
        X_temp = pd.DataFrame(scaler.inverse_transform(X), columns=features)
    
    y = np.ones((np.shape(X_temp)[0], 1))
    for i in range(0,np.shape(X_temp)[0]):
        traj = solve_ivp(eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], t_eval = None, events = [event_1, event_2])
        y[i] = traj.y[1][-1]
        
    return y


def aggregate(values):
    """
        Performs data aggregation. Aggregates for each feature its contribution to
        each output variable for a given model.

        Returns
        -------
        agg_vals : pandas.DataFrame, aggregated feature importance.
    """
    feature_agg = np.empty((len(features), len(labels)))
    for i, label in enumerate(labels):
        for j, feature in enumerate(features):
            feature_agg[j,i] = np.mean(np.abs(values[i][:,j]))
    agg_vals = pd.DataFrame(feature_agg, columns = labels, index=features)

    return agg_vals

def vals_to_df(values, data, save=False, explainer = "lime", suffix = None):
    xt_atts = values[0]
    vt_atts = values[1]
    data = data
    
    param_array = np.array(np.ones((data.shape[0], 5)))
    for i, param in enumerate(parameters):
        param_array[:,i] = parameters[param]*np.ones((data.shape[0]))

    df = pd.DataFrame(xt_atts, columns = ["xt_x0", "xt_v0", "xt_t", "xt_rand"])
    df = df.join(pd.DataFrame(vt_atts, columns = ["vt_x0", "vt_v0", "vt_t", "vt_rand"]))
    df = df.join(pd.DataFrame(data.values, columns = features))
    df = df.join(pd.DataFrame(param_array, columns = parameters.keys()))
    df.insert(1, 'explainer' ,[explainer for i in range(df.shape[0])])
    
    if save:
        df.to_csv("Results/"+explainer+"/"+explainer+"_vals_"+suffix+".csv")
    return df

class Bootstrapper():
    def __init__(model, data, explainer_type = 'kernel', num_straps = 50, back_size = 100, 
                 features = features, labels = labels, lime_models = [true_lime_1, true_lime_2], suffix = suffix)
        self.explainer_type = explainer_type
        self.model = model
        self.data = data
        self.features = features
        self.labels = labels
        self.num_straps = num_straps
        self.back_size = back_size
        self.suffix = suffix
        
    def bootstrap(self, X):
        self.values = np.empty((self.num_straps, len(self.labels), len(self.features)))
        self.mean_std_arr = np.empty((2, len(self.labels), len(self.features)))
        for i in range(self.num_straps):
            background_i = shap.sample(self.data, self.back_size, random_state = np.random.randint(100))
            if self.explainer_type == 'kernel':
                exp_i = shap.KernelExplainer(self.model, background_i)
                shapper = exp_i.shap_values(X)
            elif self.explainer_type == 'sample':
                exp_i = shap.SampleExplainer(self.model, background_i)
                shapper = exp_i.shap_values(X)
            else self.explainer_type == 'lime':
                exp_i = MyLime(self.lime_models[0], self.lime_models[1], background_i, mode='regression')
                shapper = exp_i.attributions(X)
            self.values[i,0,:] = shapper[0]
            self.values[i,1,:] = shapper[1]
        for i in range(len(labels)):
            for j in range(len(features)):
                self.mean_std_arr[0, i, j] = np.mean(self.values[:,i,j])
                self.mean_std_arr[1, i, j] = np.std(self.values[:,i,j])
            
    return self.mean_std_arr
    
    def to_df(self):
        self.bootstrap_df = self.x_list.copy()
        for k, col in enumerate(["mean", "std"]):
            for j in range(len(self.labels)):
                for i in range(len(self.features)):
                    self.bootstrap_df.insert(4 + i + j*len(self.features) + k*len(self.features)*len(self.labels), 
                                             self.features[i] + "_" + self.labels[j] + "_" + col, 
                                             self.bootstrap_array[:,k,j,i])
        if self.save:
            self.bootstrap_df.to_csv("Results/"+self.explainer_type+"/"+self.explainer_type+"bootstrap_vals_"+self.suffix+".csv")
        return self.bootstrap_df
    def calculate(self, num_samples = 50, save = True):
        self.x_list = self.data.iloc[np.sort(np.random.choice(self.data.shape[0], num_samples, replace =False))]
        self.bootstrap_array = np.empty((num_strapped_samples, 2, len(labels), len(features)))
        for i in range(x_list.shape[0]):
            x_val = self.x_list.iloc[i,:]
            self.bootstrap_array[i,:,:,:] = self.bootstrap(x_val)
        return self.to_df()
    
    
    
class MyLime(shap.other.LimeTabular):
    def __init__(self, model, flipped_model, data, mode="classification"):
        self.model = model
        self.flipped = flipped_model
        assert mode in ["classification", "regression"]
        self.mode = mode

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(data, mode=mode)
        
        out = self.model(data[0:1])
        flipped_out = self.flipped(data[0:1])
        self.out_dim = self.model(data[0:1]).shape[1]
        self.flat_out = False
            
    def attributions(self, X, nsamples=5000, num_features=None):
        num_features = X.shape[1] if num_features is None else num_features
        
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(self.out_dim+1)]
        for i in tqdm(range(X.shape[0]), desc="Generating Data…", ascii=False, ncols=75):
        #for i in range(X.shape[0]):
            for j in range(self.out_dim):
                    exp1 = self.explainer.explain_instance(X[i], self.model, labels=range(self.out_dim), 
                                                           num_features=num_features, num_samples=500)
                exp2 = self.explainer.explain_instance(X[i], self.flipped, labels=range(self.out_dim), 
                                                       num_features=num_features, num_samples=500)
                for k, v in exp1.local_exp[1]:
                    #print("boo")
                    out[0][i,k] = v
                for k, v in exp2.local_exp[1]:
                    #print("boo")
                    out[1][i,k] = v
          
        return out