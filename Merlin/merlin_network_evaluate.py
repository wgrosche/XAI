# import libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft

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


class Duffing():
    """
        Class for the Duffing Oscillator
    """
    def __init__(self, parameters = {'alpha': 0.3, 'beta': -0.1, 'gamma': 0.37, 'delta': 0.3, 'omega': 1.2}, 
                 labels = ['xt','vt'], features = ['x0','v0', 't', 'rand', 'gamma', 'f_of_x'], scaler = None, f_of_x = None, time = True, num_gammas = 1):
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
        self.parameters = parameters
        self.labels = labels
        self.features = features
        self.scaler = scaler
        self.suffix = "params_"+str(parameters['alpha'])+"_"+str(parameters['beta'])+"_"+str(parameters['gamma'])+"_"+str(parameters['delta'])+"_"+str(parameters['omega'])
        self.num_gammas = num_gammas
        self.f_of_x = f_of_x
        self.model_type = '_all'
        

            
        
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

    def termination_event(self, t, y):
        """
            Stops Numerical Integration once points wander too far away
        """
        return (np.abs(y[0]) - 10)*(np.abs(y[1]) - 10)
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
        if num_gammas > 1:
            self.num_gammas = num_gammas
            self.suffix = self.suffix + '_gamma_on'
            self.features = self.features + ['gamma']
        self.scaler = None
        #Define bounds of the sampling
        x_min = -2
        x_max = 2
        v_min = -2
        v_max = 2
        #Initialise the output arrays        
        X = np.empty((num_samples*samples*self.num_gammas, len(np.hstack((self.features, self.labels)))))
        #Define the t_range to draw from
        t_range = np.linspace(0, end_time, 100, endpoint=False)
        t_vals = np.sort(np.random.choice(t_range, size = samples, replace=False))

        #Generate num_samples samples
        for i in tqdm(range(num_samples), desc="Generating Data…", ascii=False, ncols=75):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min
            for j in range(self.num_gammas):
                if self.num_gammas > 1:
                    self.parameters['gamma'] = np.random.random_sample()
                #Generate a trajectory
                trajectory = solve_ivp(self.eom, [0, end_time], [x0,v0], t_eval = t_vals, events = [self.termination_event])
                traj_cutoff =  samples - len(trajectory.y[0])
                if traj_cutoff > 0:
                    trajectory.y[0] = np.append(trajectory.y[0].reshape(-1,1), 10.0*np.ones(traj_cutoff))
                    trajectory.y[1] = np.append(trajectory.y[1].reshape(-1,1), 10.0*np.ones(traj_cutoff))

                val_range_low = i*samples*self.num_gammas+j*samples
                val_range_high = i*samples*self.num_gammas+(j+1)*samples
                X[val_range_low:val_range_high,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                               v0*np.ones(samples).reshape(-1,1),
                                               t_vals.reshape(-1,1),
                                               np.random.uniform(-1,1,samples).reshape(-1,1),
                                               self.parameters['gamma']*np.ones(samples).reshape(-1,1),
                                               self.f_of_x(x0,v0)*np.ones(samples).reshape(-1,1),
                                               trajectory.y[0].reshape(-1,1), 
                                               trajectory.y[1].reshape(-1,1)))

        self.X_df = pd.DataFrame(X, columns = np.hstack((self.features, self.labels)))
        return self.X_df

    def scale_features(self):
        if self.scaler == None:
            self.scaler = MinMaxScaler(feature_range=[0,1])
            self.X_df[self.features] = self.scaler.fit_transform(self.X_df[self.features].values)
        else: return
    def energy(self, x, v):
        return 0.5*v**2 + 0.5*self.parameters['alpha']*x**2 +0.25*self.parameters['beta']*x**4
        

        
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
alpha = 1.0
beta = 1.0
gamma = 0.37
delta = 1.0
omega = 1.2

def energy(x, v):
    return 0.5*v**2 + 0.5*alpha*x**2 +0.25*beta*x**4

parameters_now = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'omega':omega}
duffing = Duffing(parameters = parameters_now, num_gammas = 20, f_of_x = energy)
eom = duffing.eom
suffix = duffing.suffix
model_type = duffing.model_type

end_time = 100
duffing.generate(100, samples = 100, end_time = end_time)
duffing.scale_features()
X = duffing.X_df[duffing.features]
y = duffing.X_df[duffing.labels]


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

# Create a basic model instance
model = MLModel()

"""
Train Model
"""
#model.build()
# Display the model's architecture
#model.summary()



# Create a callback that saves the model's weights
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25),
             tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25)]


# pipe = make_pipeline(scaler, model)

history=model.fit(X_train, y_train, steps_per_epoch=None, epochs=500, 
                  validation_split=0.2, batch_size=1024, shuffle=True, callbacks=callbacks, verbose=1)

loss = model.evaluate(X_test, y_test, verbose=1)
print("Trained model, loss: {:5.2f}%".format(loss))

model.save("Models/gamma_ml_model_"+suffix+model_type)

def ml_x(X):
    return model.predict(X)[0]
def ml_v(X):
    return model.predict(X)[1]

explainers = ["kernel", "sampling", "lime", "numeric"]
true_lime = [duffing.predict_x, duffing.predict_v]
ml_lime = [ml_x, ml_v]
models = {"true" : duffing.predict, "ml" : model.predict}
lime_models = {"true" : true_lime, "ml" : ml_lime}


background = shap.sample(X_test, 100)
choice = X.iloc[np.sort(np.random.choice(X_test.shape[0], 100, replace =False))]


big_df = pd.DataFrame()
for model in models:
    for explainer in explainers:
        print(explainer + model)
        if explainer == "kernel":
            temp_explainer = shap.KernelExplainer(models[model], background)
            temp_vals = temp_explainer.shap_values(choice)
        elif explainer == "sampling":
            temp_explainer = shap.SamplingExplainer(models[model], background)
            temp_vals = temp_explainer.shap_values(choice)
        elif explainer == "lime":
            temp_explainer = MyLime(lime_models[model], choice, mode='regression')
            temp_vals = temp_explainer.attributions(choice)
        elif explainer == "numeric":
            temp_explainer = NumericExplainer(models[model], duffing.features, duffing.labels, h = 0.001)
            temp_vals = temp_explainer.feature_att(choice)
        else:
            print("not a valid explainer type")
        big_df = big_df.append(duffing.vals_to_df(temp_vals, 
                                                        choice, save=False, explainer = explainer, suffix = suffix+model))
        
        
big_df.to_csv("Results/explainer_dataframe_"+suffix+model_type+".csv")  

