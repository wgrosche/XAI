# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:06:21 2021

@author: wilke
"""

# General Libraries
import numpy as np
import pandas as pd
import os

# True Model
from scipy.integrate import odeint
from scipy.fft import fft

# Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Sequence
from tensorflow import keras


# Data Preprocessing
from sklearn.preprocessing import StandardScaler





# My Modules
from wilkeXAI.data_generator import DataGenerator
import wilkeXAI.wilke_shap as fwg
    
"""
Define and Machine Learning Model
"""

def MLModel():
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

    
def main():
    # Define name for this run
    suffix = "Run_1"
    generator = DataGenerator()
    # Generate the data
    X_train, y_train = generator.generate(num_samples=int(1e5))
    X_test, y_test = generator.generate(num_samples=int(1e3))
    
    # Scale the Data
    scaler = StandardScaler()

    scaler.fit(X_train.values)
    scaler.transform(X_train.values, copy=False)
    scaler.transform(X_test.values, copy = False)
    
    model = MLModel()
    true_model = fwg.TrueModel(scaler, X_test)
    
    # Train Network
    # Model Weights Path
    checkpoint_path = "Networks/training/"+suffix+"cp1.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    callbacks = [cp_callback,
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                 tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)]


    history=model.fit(X_train, y_train, steps_per_epoch=None, epochs=500, 
                      validation_split=0.2, batch_size=20364, shuffle=True, callbacks=callbacks, verbose=1)
    
    models = {'ml': model, 
         'true': true_model}
       
    for exp_type in ['shap', 'lime', 'analytic']:
        explainer_curr = fwg.wilke_explainer(models, X_train, X_test, y_test, explainer_type=exp_type)
        explainer_curr.eval_explainer().to_csv("Results/"+exp_type+"/individual/"+suffix+".csv")
        explainer_curr.aggregate().to_csv("Results/"+exp_type+"/aggregate/"+suffix+".csv")

    



if __name__ == "__main__":
    main()
