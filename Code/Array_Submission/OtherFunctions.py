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
        self.f = f.predict
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
    
    
    
    
class Bootstrapper():
    """
    Function to perform bootstrapping of feature importances.
    Takes model and a dataset and applies an explainer to the model.
    Calculates confidence intervals by bootstrapping.
    """
    def __init__(self, model, data, features, labels, suffix, explainer_type, num_straps = 50, back_size = 100):
        """
        Initialisation
        
        Parameters:
        
        model: model to be evaluated with explainability methods, f: features -> labels
        data: dataset to be used as the background dataset for the explainers (2d array(features, n))
        features: data features (list)
        labels: data labels (list)
        suffix: str to use when saving the results
        
        """
        self.explainer_type = explainer_type
        self.model = model
        self.data = data
        self.features = features
        self.labels = labels
        self.num_straps = num_straps
        self.back_size = back_size
        self.suffix = suffix
        
    def bootstrap(self, X):
        """
        Function that performs the bootstrapping. 
        Randomly changes the background dataset and records the new attributions
        Parameters:
        X : instances to explain
        Returns:
        array of mean feature attributions for the samples in X
        """
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
            elif self.explainer_type == 'lime':
                exp_i = MyLime(self.model, background_i, mode="regression")
                shapper = exp_i.attributions(X)
            self.values[i,0,:] = shapper[0]
            self.values[i,1,:] = shapper[1]
        for i in range(len(self.labels)):
            for j in range(len(self.features)):
                self.mean_std_arr[0, i, j] = np.mean(self.values[:,i,j])
                self.mean_std_arr[1, i, j] = np.std(self.values[:,i,j])
            
        return self.mean_std_arr
    
    def to_df(self):
        """
        transforms the bootstrap results to a dataframe
        saves the bootstrapped samples dataframe
        """
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
    def calculate(self, num_samples = 10, save = True):
        self.x_list = self.data.iloc[np.sort(np.random.choice(self.data.shape[0], num_samples, replace =False))]
        self.bootstrap_array = np.empty((num_samples, 2, len(self.labels), len(self.features)))
        for i in tqdm(range(self.x_list.shape[0]), desc="Bootstrapping…", ascii=False, ncols=75):
            x_val = self.x_list.iloc[i,:]
            self.bootstrap_array[i,:,:,:] = self.bootstrap(x_val)
        return self.to_df()
    
    
class MyLime(shap.other.LimeTabular):
    """
    Implementation of LIME tabular to allow use with the models used in this work.
    """
    def __init__(self, model, data, mode="classification", discretize_continuous = False, kernel_width = None):
        """
        Initialisation
        
        Parameters:
        model : model to evaluate with LIME, either tensorflow NN or function: vector -> scalar
        data : background dataset to be used with LIME for the purposes of scaling feature perturbations
        mode : only regression should be used in this case. defines the mode to use with LIME
        discretize continuous : passes the setting for data discretization to LimeTabularExplainer
        """
        self.model = model
        assert mode in ["classification", "regression"]
        self.mode = mode

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(data, mode=mode, discretize_continuous=discretize_continuous, kernel_width = kernel_width)
        self.out_dim = 1#self.model(data[0:1]).shape[1]
            
    def attributions(self, X, num_samples=5000):
        """
        Feature attributions made my LIME
        
        Parameters:
        X : data for which the feature attributions are to be calculated, 2d array
        num_samples : number of samples to be passed through the model to generate
                      labels for the linear surrogate model
        Returns:
        Feature attributions made by LIME: array of shape (num_labels, num_features, length X)
        """
        try:
            num_features = X.shape[1] 
        except:
            print('exception')
            num_features = 1
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(len(self.model))]
        for i in tqdm(range(X.shape[0]), desc="Calculating Lime…", ascii=False, ncols=75):
            exp1 = self.explainer.explain_instance(X[i], self.model[0], labels=range(self.out_dim), 
                                                    num_features=num_features, num_samples=num_samples)
            exp2 = self.explainer.explain_instance(X[i], self.model[1], labels=range(self.out_dim), 
                                                    num_features=num_features, num_samples=num_samples)
            for k, v in exp1.local_exp[1]: 
                out[0][i,k] = v
            for k, v in exp2.local_exp[1]: 
                out[1][i,k] = v
          
        return out
    
    
    
def MLModel():
    """
    The DNN or "complex" model.
    """
    opt = Adam(learning_rate=0.001, beta_1=0.7)
    loss='mse'
    model = Sequential([
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(32, activation='tanh'),
        layers.Dense(2)            
    ])
    model.compile(optimizer=opt, loss=loss)
    return model
  
    
def SimpleModel():
    """
    The SNN or "simple" model.
    """
    opt = Adam(learning_rate=0.001, beta_1=0.7)
    loss='mse'
    model = Sequential([
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)            
    ])
    model.compile(optimizer=opt, loss=loss)
    return model
