import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV




suffix = None


# Load Datasets
X_train = pd.read_csv("X_train"+suffix+".csv", header=0, index_col=0)
y_train = pd.read_csv("y_train"+suffix+".csv", header=0, index_col=0)





# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model():
    model = Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))#old 6*32
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(2))
    model.compile()
    return model

def main():

    # wrap the model using the function
    clf = KerasRegressor(build_fn=create_model, verbose=0)

    # create parameter grid, as usual, but note that you can
    # vary other model parameters such as 'epochs' (and others 
    # such as 'batch_size' too)
    param_grid = {
        'clf__steps_per_epoch':[100],
        'clf__loss':['mse','mae']
        'clf__optimizer':['rmsprop','adam','adagrad', Adam(learning_rate=0.001, beta_1=0.7)],
        'clf__epochs':[100],#,200,500,1000,2000],
        'clf__dropout':[0.1,0.2],
        'clf__kernel_initializer':['glorot_uniform','normal','uniform'],
        'clf__batch_size':[128,256,512,1024,2048],
        'clf__validation_split':[0.05, 0.1],
        'clf__shuffle':[True, False],
        'clf__callbacks':[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                          tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250),
                          tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15),
                          tf.keras.callbacks.EarlyStopping(monitor='loss', patience=250),
                          [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250),
                          tf.keras.callbacks.EarlyStopping(monitor='loss', patience=250)],
                          [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25),
                          tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25)]]
    }


    # just create the pipeline
    pipeline = Pipeline([
        ('preprocess', StandardScaler()),
        ('clf', clf) 
    ])


    # if you're not using a GPU, you can set n_jobs to something other than 1
    grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    grid.fit(X_train, y_train)

    pd.DataFrame(grid.cv_results).to_csv("grid_optimisation.csv")
    
if __name__ == "__main__":
    main()