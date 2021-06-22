"""
Import Libraries
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# Explainability
import shap


"""Function Definitions"""

"""
Define and Create Model
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


class TrueModel():
    """
    Class to represent the True Model of the Duffing oscillator.
    Uses the scipy odeint integrator to perform time evolution.
    
    Methods:
    -----------
    predict(X):
        Inputs:
        --------
        X: pandas DataFrame with at least columns x0,v0,t
        Returns:
        --------
        y: pandas DataFrame with columns xt,vt
    """
    def __init__(self, scaler):
        self.alpha=-1
        self.beta=1
        self.delta=0.3
        self.gamma=0.37
        self.omega=1.2
        self.scaler = scaler
        
    def eom(self, u, t):
        x, dx = u[0], u[1]
        ddx= self.gamma * np.cos(self.omega * t) - (self.delta * dx + self.alpha*x + self.beta * x**3)
        return [dx,ddx]
    
    def predict(self, X):
        X = pd.DataFrame(self.scaler.inverse_transform(X), columns=['x0','v0','t','r'])
        y = np.ones((np.shape(X)[0], 2))
        for i in range(0,np.shape(X)[0]):
            t_range = np.linspace(0, X['t'].iloc[i], 500, endpoint=False)
            y[i,:] = odeint(self.eom, [X['x0'].iloc[i],X['v0'].iloc[i]], t_range)[-1]
        y = pd.DataFrame(y, columns=['xt','vt'])    
        return y

def Remove_Outlier_Indices(df):
    Q1 = df.quantile(0.00)
    Q3 = df.quantile(0.95)
    IQR = Q3 - Q1
    trueList = ~((df > (Q3 + 1.5 * IQR)))
    #trueList = ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))
    return trueList



def main():
    """
    Choose the training data set to be trained on.

    For general datasets use custom suffix.
    """
    suffix = "delta03_irrelevant"#"delayed_traj_delta03"

    X_train = pd.read_csv("Data/X_train_"+suffix+".csv", header=0, index_col=0)
    y_train = pd.read_csv("Data/y_train_"+suffix+".csv", header=0, index_col=0)
    X_test = pd.read_csv("Data/X_test_"+suffix+".csv", header=0, index_col=0)
    y_test = pd.read_csv("Data/y_test_"+suffix+".csv", header=0, index_col=0)


    scaler = StandardScaler()

    # Model Weights Path
    checkpoint_path = "Networks/training"+suffix+"cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    scaler.fit(X_train.values)
    scaler.transform(X_train.values, copy=False)
    scaler.transform(X_test.values, copy = False)

    model = MLModel()
    true_model = TrueModel(scaler)

    """
    Train Model
    """
    #model.build()
    # Display the model's architecture
    #model.summary()



    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    callbacks = [cp_callback,
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]


    history=model.fit(X_train, y_train, steps_per_epoch=None, epochs=500, 
                    validation_split=0.2, batch_size=20364, shuffle=True, callbacks=callbacks, verbose=1)




    background = shap.sample(X_train, 1000)

    kernel_model = shap.KernelExplainer(model.predict, background)
    kernel_true = shap.KernelExplainer(true_model.predict, background)

    vals = np.abs(np.linalg.norm(model.predict(X_test), axis=1)- np.linalg.norm(y_test, axis=1))

    X_test_good = X_test.iloc[np.where(vals < 0.1)]


    shap_data_x0 = X_test_good.sort_values(by='x0').iloc[np.where(np.multiply(np.abs(scaler.inverse_transform(X_test_good)[:,1])<0.2,
                                                                            np.abs(scaler.inverse_transform(X_test_good)[:,2]-50)<5))]
    shap_choice_x0 = shap_data_x0.iloc[np.sort(np.random.choice(shap_data_x0.shape[0], 100, replace=False))]
    shap_choice_x0 = pd.DataFrame(shap_choice_x0.values, columns = X_test.columns)


    shap_data_v0 = X_test_good.sort_values(by='v0').iloc[np.where(np.multiply(np.abs(scaler.inverse_transform(X_test_good)[:,0])<0.5,
                                                                            np.abs(scaler.inverse_transform(X_test_good)[:,2]-50)<5))]
    shap_choice_v0 = shap_data_v0.iloc[np.sort(np.random.choice(shap_data_v0.shape[0], 100, replace=False))]
    shap_choice_v0 = pd.DataFrame(shap_choice_v0.values, columns = X_test.columns)


    shap_data_t = X_test_good.sort_values(by='t').iloc[np.where(np.multiply(np.abs(scaler.inverse_transform(X_test_good)[:,1])<0.2,
                                                                            np.abs(scaler.inverse_transform(X_test_good)[:,0])<0.5))]
    shap_choice_t = shap_data_t.iloc[np.sort(np.random.choice(shap_data_t.shape[0], 100, replace=False))]
    shap_choice_t = pd.DataFrame(shap_choice_t.values, columns = X_test.columns)

    sorted_true_values_x0 = kernel_true.shap_values(shap_choice_x0)
    sorted_model_values_x0 = kernel_model.shap_values(shap_choice_x0)

    sorted_true_values_v0 = kernel_true.shap_values(shap_choice_v0)
    sorted_model_values_v0 = kernel_model.shap_values(shap_choice_v0)

    sorted_true_values_t = kernel_true.shap_values(shap_choice_t)
    sorted_model_values_t = kernel_model.shap_values(shap_choice_t)

    true_plot_data_x0_xt = pd.DataFrame(sorted_true_values_x0[0], columns = X_test.columns)

    true_plot_data_v0_xt = pd.DataFrame(sorted_true_values_v0[0], columns = X_test.columns)

    true_plot_data_t_xt = pd.DataFrame(sorted_true_values_t[0], columns = X_test.columns)

    true_plot_data_x0_vt = pd.DataFrame(sorted_true_values_x0[1], columns = X_test.columns)

    true_plot_data_v0_vt = pd.DataFrame(sorted_true_values_v0[1], columns = X_test.columns)

    true_plot_data_t_vt = pd.DataFrame(sorted_true_values_t[1], columns = X_test.columns)


    model_plot_data_x0_xt = pd.DataFrame(sorted_model_values_x0[0], columns = X_test.columns)

    model_plot_data_v0_xt = pd.DataFrame(sorted_model_values_v0[0], columns = X_test.columns)

    model_plot_data_t_xt = pd.DataFrame(sorted_model_values_t[0], columns = X_test.columns)

    model_plot_data_x0_vt = pd.DataFrame(sorted_model_values_x0[1], columns = X_test.columns)

    model_plot_data_v0_vt = pd.DataFrame(sorted_model_values_v0[1], columns = X_test.columns)

    model_plot_data_t_vt = pd.DataFrame(sorted_model_values_t[1], columns = X_test.columns)

    true_plot_data_x0_xt.to_csv("true_plot_data_x0_xt.csv")

    true_plot_data_v0_xt.to_csv("true_plot_data_v0_xt.csv")

    true_plot_data_t_xt.to_csv("true_plot_data_t_xt.csv")

    true_plot_data_x0_vt.to_csv("true_plot_data_x0_vt.csv")

    true_plot_data_v0_vt.to_csv("true_plot_data_v0_vt.csv")

    true_plot_data_t_vt.to_csv("true_plot_data_t_vt.csv")


    model_plot_data_x0_xt.to_csv("model_plot_data_x0_xt.csv")

    model_plot_data_v0_xt.to_csv("model_plot_data_v0_xt.csv")

    model_plot_data_t_xt.to_csv("model_plot_data_t_xt.csv")

    model_plot_data_x0_vt.to_csv("model_plot_data_x0_vt.csv")

    model_plot_data_v0_vt.to_csv("model_plot_data_v0_vt.csv")

    model_plot_data_t_vt.to_csv("model_plot_data_t_vt.csv")


if __name__ == "__main__":
    main()