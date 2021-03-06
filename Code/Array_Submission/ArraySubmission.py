# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import sys

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

# read in parameter configuration from the batch commit
idx = int(sys.argv[1])
model_setting = sys.argv[2]
feature_setting = sys.argv[3]

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

parameter_list = [{'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2}, 
                  {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 1.0, 'omega' : 1.2},
                  {'alpha' : 1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2}, 
                  {'alpha' : -1.0, 'beta' : -1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 1.2},
                  {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.37, 'delta' : 0.3, 'omega' : 0.1},
                  {'alpha' : -1.0, 'beta' : 1.0, 'gamma' : 0.5, 'delta' : 0.3, 'omega' : 1.2}]

dict_param = parameter_list[idx]

from OtherFunctions import *
if feature_setting == "Base":
    num_samples_ml = 100000
    from  BaseDuffing import Duffing
elif feature_setting == "Random":
    num_samples_ml = 100000
    from  RandomDuffing import Duffing
elif feature_setting == "Energy":
    num_samples_ml = 100000
    from  EnergyDuffing import Duffing
elif feature_setting == "Gamma":
    num_samples_ml = 1000
    from  GammaDuffing import Duffing
    


if __name__ == '__main__':
    # initialise the duffing oscillator in the parameter configuration of choice
    duffing = Duffing(parameters = dict_param)
    # define the suffix to use when saving objects
    suffix = feature_setting + "_" + model_setting + "_" + duffing.suffix
    # generate training and test samples
    end_time = 100
    duffing.generate(num_samples_ml, samples = 100, end_time = end_time) #samples prev 100
    duffing.scale_features()
    X_train, X_test, y_train, y_test = train_test_split(duffing.X_df[duffing.features], 
                                                        duffing.X_df[duffing.labels], test_size=0.1, random_state=42)
    
    
    # Create a basic model instance
    if model_setting == "Complex":
        model = MLModel()
    elif model_setting == "Simple":
        model = SimpleModel()
    elif model_setting == "True":
        model = duffing
        
    
    if (model_setting == "Simple") or (model_setting == "Complex"):
        """
        Train Model if the model is a NN
        """
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3),
                     tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]


        history=model.fit(X_train, y_train, steps_per_epoch=None, epochs=500, validation_split=0.2, 
                          batch_size=1024, shuffle=True, callbacks=callbacks, verbose=0)


        model.save('Models/Model'+suffix)
        with open('Models/TrainingHistory/'+suffix, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    # since some explainers require the function to be 
    # either NN or function we define model_ 
    # to implement predict for duffing
    if model_setting == "Complex":
        model_ = model
    elif model_setting == "Simple":
        model_ = model
    elif model_setting == "True":
        model_ = duffing.predict
        
    """
    If models are already trained and just need to be loaded 
    comment out the training block above and comment in this block:
    if model_setting == "Complex":
        model = tf.keras.models.load_model("Models/Model"+suffix)
        model_ = model
    elif model_setting == "Simple":
        model = tf.keras.models.load_model("Models/Model"+suffix)
        model_ = model
    elif model_setting == "True":
        model = duffing
        model_ = duffing.predict
    """    
    """
    My implementation of LIME requires each label to be predicted individually.
    This is somewhat inefficient but we use this hack to facilitate this.
    """
    def lime_x(X):
        return model.predict(X)[:,0]
    def lime_v(X):
        return model.predict(X)[:,1]
    
    lime_models = [lime_x, lime_v]
    # list of explainers to apply to the models
    explainers = ["kernel", "sampling", "lime", "numeric"]
    
    # define background sample to use when evaluating SHAP and LIME
    background = shap.sample(X_test, 100)
    # choose samples on which to evaluate SHAP And LIME
    choice = X_test.iloc[np.sort(np.random.choice(X_test.shape[0], 100, replace =False))]

    
    big_df = pd.DataFrame()
    for explainer in explainers:
        if explainer == "kernel":
            temp_explainer = shap.KernelExplainer(model_, background)
            temp_vals = temp_explainer.shap_values(choice)
        elif explainer == "sampling":
            temp_explainer = shap.SamplingExplainer(model_, background)
            temp_vals = temp_explainer.shap_values(choice)
        elif explainer == "lime":
            temp_explainer = MyLime(lime_models, X_test, mode='regression', discretize_continuous = False)
            temp_vals = temp_explainer.attributions(choice)
        elif explainer == "numeric":
            temp_explainer = NumericExplainer(model, duffing.features, duffing.labels, h = 0.001)
            temp_vals = temp_explainer.feature_att(choice)
        else:
            print("not a valid explainer type")
        # save the explanations along with the features that led to them
        big_df = big_df.append(duffing.vals_to_df(temp_vals, choice, explainer = explainer, suffix = suffix))


    big_df.to_csv("Results/explainer_dataframe_"+suffix+".csv")  

