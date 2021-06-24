# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:03:54 2021

@author: wilke

Explainer class for the XAI master thesis
"""

"""
Import Libraries
"""
# General Libraries
import numpy as np
import pandas as pd

# True Model
from scipy.integrate import odeint

# Plotting Libraries
import matplotlib.pylab as plt
import seaborn as sns

# Explainability
import lime
import shap


class AnalyticExplainer():
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
            if first_run:
                self.__atts = grads.transpose()
                first_run = False
            else:
                self.__atts = [self.__atts, grads.transpose()]
                        
        return self.__atts
    



class wilke_explainer():
    """
        Class to evaluate models and plot both the individual and aggregate 
        feature importance. Implements a method for shap and for lime.
        Implements methods from the SHAP library, tailored for this notebook.
    """
    def __init__(self, models, background_data, test_data, test_labels, suffix = None, data_tol=0.1, num_vals=100, 
                 explainer_type='shap', background_resolution=100, tolerance=0.1):
        """
            Initialises the explainer objects for the various models.
            
            Parameters
            ----------
            models : dictionary of models that implement a predict function
                that take a 2d numpy.array or pandas.DataFrame
                
            background_data : pandas.DataFrame with samples of training data,
                used to create a masker for the explainer objects
                
            test_data : pandas.DataFrame with samples of test data,
                used to choose samples on which to evaluate the explainers.
                
            test_labels : pandas.DataFrame with samples of test labels,
                used to choose good samples on which to evaluate the explainers.

            suffix : str, added to the saving filepath for the plots.
                
            data_tol : float, thickness of band in which data_points are considered 
                to be close to the mean value.
                
            num_vals : int, maximum number of values on which the explainers are
                evaluated.
                
            explainer_type : str, 'shap' or 'lime', decides which explainer is being
                implemented: shap.KernelExplainer or shap.LimeTabular.
                
            background_resolution : int, number of samples of the background_data to
                be used to create the masker for the explainers.
                
            tolerance : float, determines how close a model prediction must be to the
                true value for it to be considered a good estimate.

        """
        self.num_vals = num_vals
        self.tol = data_tol
        self.models = models
        self.background_data = background_data
        self.test_data = test_data
        self.test_labels = test_labels
        self.tolerance = tolerance
        self.explainers = {}
        self.suffix = suffix
        self.background = shap.sample(self.background_data, background_resolution)
        self.explainer_type = explainer_type
        if explainer_type=='shap':
             for mod in models:
                self.explainers[mod] = shap.KernelExplainer((models[mod]).predict, self.background)
        if explainer_type=='lime':
             for mod in models:
                self.explainers[mod] = shap.other.LimeTabular((models[mod]).predict, 
                                                                     self.background, mode="regression")
                
        if explainer_type=='analytic':
            for mod in models:
                self.explainers[mod] = AnalyticExplainer(models[mod].predict, test_data.columns, test_labels.columns)
    
    def choose_data(self, i, feature, num_features):
        """
            Chooses the data points on which the explainer will be evaluated for
            a given feature. First, removes any points where the prediction is
            further than tolerance from the true value. Second, takes values in
            a band of thickness tol around 0 for all features not currently being
            evaluated on. Third, chooses a random subset of length num_vals of 
            these values. This function should never be called outside of this class.
            
            Parameters
            ----------
            i : int, index of the feature being explained
            feature : str, the feature being explained
            num_features : the total number of features
            
            Returns
            ----------
            data_arr : pandas.DataFrame, array of the chosen values.
        """
        vals = np.abs(np.linalg.norm((self.models['ml']).predict(self.test_data), axis=1) - 
                      np.linalg.norm(self.test_labels, axis=1))
        data_arr = self.test_data.iloc[np.where(vals < self.tolerance)]
        where__ = np.ones_like(data_arr.values[:,i], dtype=bool)
        for j in range(1,num_features):
            where__ = np.multiply(where__, np.abs(data_arr.values[:,(i + j)%num_features])<self.tol)
        data_arr = data_arr.iloc[where__]
        data_arr = data_arr.iloc[np.sort(
            np.random.choice(data_arr.shape[0], np.min([self.num_vals, data_arr.shape[0]]), replace=False))]
        return data_arr
        
    def eval_explainer(self):
        """
            Evaluates the explainers on data chosen by the choose_data function.
            
            Returns
            ---------
            feature_attributions : pandas.DataFrame, array of the feature attributions
                with a multiindex (index, feature, contribution, model)
        """
        first_run = True
        for i, __feature in enumerate(self.test_data.columns):
            arr = self.choose_data(i, __feature, len(self.test_data.columns))
            for __explainer in self.explainers:
                if self.explainer_type=='shap':
                    __atts = self.explainers[__explainer].shap_values(arr)
                if self.explainer_type=='lime':
                    __atts = self.explainers[__explainer].attributions(arr)
                    
                if self.explainer_type=='analytic':
                    __atts = self.explainers[__explainer].feature_att(arr)
                    
                for j, __contribution in enumerate(self.test_labels.columns):
                    multi_index = [range(len(arr)), [__feature for i in range(len(arr))], 
                                   [__contribution for i in range(len(arr))],
                                   [__explainer for i in range(len(arr))]]
                    if first_run:
                        self.feature_attributions = pd.DataFrame(__atts[j], 
                                                                 columns = self.test_data.columns, 
                                                                 index = pd.MultiIndex.from_arrays(multi_index, 
                                                                        names=('num', 'feature', 'contribution', 'model')))
                    else:
                        self.feature_attributions = self.feature_attributions.append(pd.DataFrame(__atts[j], 
                                                                 columns = self.test_data.columns, 
                                                                 index = pd.MultiIndex.from_arrays(multi_index, 
                                                                        names=('num', 'feature', 'contribution', 'model'))))
                    first_run = False
                    
        return self.feature_attributions
        
    def exp_plot(self):
        """
            Plotting routine to visualise the explainers' results. Plots individual
            feature contribution for each model and each feature.
        """
        f, axs = plt.subplots(self.test_labels.shape[1], self.test_data.shape[1], 
                              figsize=(4*self.test_data.shape[1], 8), 
                              gridspec_kw=dict(width_ratios=4*np.ones((self.test_data.shape[1]))))

        for i, __feature in enumerate(self.test_data.columns):
            for j, __contribution in enumerate(self.test_labels.columns):
                for __model in self.models:
                    sns.scatterplot(data = self.feature_attributions.xs((__feature, __contribution, __model), 
                                                      level=('feature', 'contribution', 'model')), 
                                    x = self.feature_attributions.xs((__feature, __contribution, 'true'), 
                                                   level=('feature', 'contribution', 'model')).index,
                                    y = self.feature_attributions.xs((__feature, __contribution, __model), 
                                                 level=('feature', 'contribution', 'model'))[__feature],
                                    label = __model, ax=axs[j,i])  
                    
                axs[j,i].set_title(r"Feature Contribution of "+__feature+" to "+__contribution+"")
                axs[j,i].set_xlabel('Index [ ]')
                axs[j,i].set_ylabel('Feature Contribution [ ]')

        f.tight_layout()

        f.savefig("Images/"+self.explainer_type+"_summary"+self.suffix+"_kernel_good.svg", dpi='figure')
    
    def agg_func(self, X):
        """
            Function by which data aggregation is performed.
        
            Inputs
            -------
            X : numpy.array, array of values for which the aggregation is performed.
                Is a list of the values for a single feature, single contribution
                and single model.
                
            Returns
            -------
            y : value of the aggregation f(X).
        """
        return np.mean(np.abs(X))
    
    def aggregate(self):
        """
            Performs data aggregation. Aggregates for each feature its contribution to
            each output variable for a given model.
            
            Returns
            -------
            agg_vals : pandas.DataFrame, aggregated feature importance.
        """
        first_run = True
        
        for i, __contribution in enumerate(self.test_labels.columns):
            for j, __model in enumerate(self.models):
                for k, __feature in enumerate(self.test_data.columns):
                    
                    multi_index = [[__feature], [__contribution], [__model]]
                    if first_run:
                        self.agg_vals = pd.DataFrame(self.agg_func(
                            self.feature_attributions.xs((__feature,
                                                          __contribution,
                                                          __model), 
                                                         level=('feature', 'contribution', 'model'))[__feature].values),
                                                     columns = ['contrib'], 
                                                     index = pd.MultiIndex.from_arrays(multi_index, 
                                                                        names=('feature', 'contribution', 'model')))
                    else:
                        self.agg_vals = self.agg_vals.append(pd.DataFrame(self.agg_func(
                            self.feature_attributions.xs((__feature,
                                                          __contribution,
                                                          __model), 
                                                         level=('feature', 'contribution', 'model'))[__feature].values),
                                                     columns = ['contrib'], 
                                                     index = pd.MultiIndex.from_arrays(multi_index, 
                                                                        names=('feature', 'contribution', 'model'))))
                    first_run = False
                    
                    
        return self.agg_vals
    
    def agg_plot(self):
        """
            Plotting routine to visualise the aggregated feature importance.
        """
        f, axs = plt.subplots(len(self.models), self.test_labels.shape[1], 
                              figsize=(6*self.test_labels.shape[1], 4*len(self.models)), 
                              gridspec_kw=dict(width_ratios=4*np.ones((self.test_labels.shape[1]))))
        
        for i, __model in enumerate(self.models):
            for j, __contribution in enumerate(self.test_labels):
                sns.barplot(data = self.agg_vals.xs((__contribution, __model), level=('contribution', 'model')),
                    x = self.agg_vals.xs((__contribution, __model), level=('contribution', 'model')).index,
                    y = 'contrib', label = __model, ax=axs[i,j])
                axs[i,j].set_title(r"Aggregate Feature Contribution to "+__contribution+" in the "+__model+" Model")
                axs[i,j].set_ylabel('Feature Contribution [ ]')

        f.tight_layout()
        f.savefig("Images/"+self.explainer_type+"_aggregated"+self.suffix+".svg", dpi='figure')


class TrueModel():
    """
    Class to represent the True Model of the Duffing oscillator.
    Uses the scipy odeint integrator to perform time evolution.
    """
    def __init__(self, scaler, X):
        """
            Intialise Model

            Inputs
            --------
            scaler : sklearn.preprocessing.Standardscaler object that has already been trained
            X : pd.DataFrame with columns giving the features
        """
        self.alpha=-1
        self.beta=1
        self.delta=0.3
        self.gamma=0.37
        self.omega=1.2
        self.scaler = scaler
        self.cols = X.columns
        
    def eom(self, u, t):
        """
            Equation of Motion for the Duffing Oscillator

            Inputs
            --------
            u : vector of floats, x, v
            t : float, time t

        """
        x, dx = u[0], u[1]
        ddx= self.gamma * np.cos(self.omega * t) - (self.delta * dx + self.alpha*x + self.beta * x**3)
        return [dx,ddx]
    
    def predict(self, X):
        """
            Calculates the temporal evolution of [X['x0'], X['v0']] to time X['t'].

            Inputs
            --------
            X : pandas DataFrame with at least columns x0,v0,t

            Returns
            --------
            y : pandas.DataFrame with columns xt,vt
        """
        X = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.cols)
        y = np.ones((np.shape(X)[0], 2))
        for i in range(0,np.shape(X)[0]):
            t_range = np.linspace(0, X['t'].iloc[i], 500, endpoint=False)
            y[i,:] = odeint(self.eom, [X['x0'].iloc[i],X['v0'].iloc[i]], t_range)[-1]
        #y = pd.DataFrame(y, columns=['xt','vt'])    
        return y
