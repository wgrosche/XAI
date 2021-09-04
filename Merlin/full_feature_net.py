# import libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

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


class Duffing():
    """
        Class for the Duffing Oscillator
    """
    def __init__(self, parameters = {'alpha': [0.3], 'beta': [-0.1], 'gamma': [0.37], 'delta': [0.3], 'omega': [1.2]}, 
                 labels = ['xt','vt'], features = ['x0','v0', 't', 'rand', 'energy'], scaler = None):
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
        self.features = features + [param for param in parameters]
        self.scaler = scaler
        self.suffix = "params_"+str(parameters['alpha'])+"_"+str(parameters['beta'])+"_"+str(parameters['gamma'])+"_"+str(parameters['delta'])+"_"+str(parameters['omega'])
        self.parameter_grid = ParameterGrid(parameters)
        self.parameters = list(self.parameter_grid)[0]
        

            
        
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
        X = np.empty((num_samples*samples*len(list(self.parameter_grid)), len(np.hstack((self.features, self.labels)))), dtype=object)
        #Define the t_range to draw from
        t_range = np.linspace(0, end_time, 100, endpoint=False)
        t_vals = np.sort(np.random.choice(t_range, size = samples, replace=False))

        #Generate num_samples samples
        for i in tqdm(range(num_samples), desc="Generating Data…", ascii=False, ncols=75):
            #Generate random starting positions
            x0 = (x_max - x_min) * np.random.random_sample() + x_min
            v0 = (v_max - v_min) * np.random.random_sample() + v_min
            for j, dict_param in enumerate(self.parameter_grid):
                # Do something with the current parameter combination in ``dict_``
                self.parameters = dict_param
                #Generate a trajectory
                trajectory = solve_ivp(self.eom, [0, end_time], [x0,v0], t_eval = t_vals, events = [self.termination_event])
                traj_cutoff =  samples - len(trajectory.y[0])
                traj_x = np.append(trajectory.y[0].reshape(-1,1), 10.0*np.ones(traj_cutoff).reshape(-1,1))
                traj_v = np.append(trajectory.y[1].reshape(-1,1), 10.0*np.ones(traj_cutoff).reshape(-1,1))
                val_range_low = i*samples*len(list(self.parameter_grid))+j*samples
                val_range_high = i*samples*len(list(self.parameter_grid))+(j+1)*samples
                X[val_range_low:val_range_high,:] = np.hstack((x0*np.ones(samples).reshape(-1,1), 
                                               v0*np.ones(samples).reshape(-1,1),
                                               t_vals.reshape(-1,1),
                                               np.random.uniform(-1,1,samples).reshape(-1,1),
                                               self.energy(x0,v0)*np.ones(samples).reshape(-1,1),
                                               np.array([[dict_param[i] for i in dict_param] for i in range(samples)]).reshape((-1,len(dict_param))),
                                               traj_x.reshape(-1,1), 
                                               traj_v.reshape(-1,1)))
        
        self.X_df = pd.DataFrame(X, columns = np.hstack((self.features, self.labels)))
        return self.X_df

    def scale_features(self):
        if self.scaler == None:
            self.scaler = MinMaxScaler(feature_range=[0,1])
            self.X_df[self.features] = self.scaler.fit_transform(self.X_df[self.features].values)
        else: return



    def predict(self, X):
        if self.scaler == None:
            self.scale_features()
        if type(X) == pd.core.frame.DataFrame:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X.values), columns=X.columns)
        elif type(X) == np.ndarray:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.features)

        y = np.ones((np.shape(X_temp)[0], 2))
        for i in range(0,np.shape(X_temp)[0]):
            for j in self.parameters:
                self.parameters[j] = X_temp[j].iloc[i]
            traj = solve_ivp(self.eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], 
                            t_eval = None, events = [self.termination_event])
            y[i] = [traj.y[0][-1], traj.y[1][-1]]
            
        return y

    def predict_x(self, X):
        if self.scaler == None:
            self.scale_features()
        if type(X) == pd.core.frame.DataFrame:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X.values), columns=X.columns)
        elif type(X) == np.ndarray:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.features)

        y = np.ones((np.shape(X_temp)[0], 1))
        for i in range(0,np.shape(X_temp)[0]):
            for j in self.parameters:
                self.parameters[j] = X_temp[j].iloc[i]
            traj = solve_ivp(self.eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], 
                            t_eval = None, events = [self.termination_event])
            y[i] = traj.y[0][-1]

        return y

    def predict_v(self, X):
        if self.scaler == None:
            self.scale_features()
        if type(X) == pd.core.frame.DataFrame:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X.values), columns=X.columns)
        elif type(X) == np.ndarray:
            X_temp = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.features)

        y = np.ones((np.shape(X_temp)[0], 1))
        for i in range(0,np.shape(X_temp)[0]):
            for j in self.parameters:
                self.parameters[j] = X_temp[j].iloc[i]
            traj = solve_ivp(self.eom, [0, X_temp['t'].iloc[i]], [X_temp['x0'].iloc[i], X_temp['v0'].iloc[i]], 
                            t_eval = None, events = [self.termination_event])
            y[i] = traj.y[1][-1]

            
        return y

        
    def vals_to_df(self, values, data, save=False, explainer = "lime", suffix = None):
        xt_atts = values[0]
        vt_atts = values[1]
        data = data
        
        param_array = np.array(np.ones((data.shape[0], 5)))
        for i, param in enumerate(self.parameters):
            param_array[:,i] = self.parameters[param]*np.ones((data.shape[0]))

        df = pd.DataFrame(xt_atts, columns = [self.labels[0] + "_" + i for i in self.features])
        df = df.join(pd.DataFrame(vt_atts, columns = [self.labels[1] + "_" + i for i in self.features]))
        df = df.join(pd.DataFrame(data.values, columns = self.features))
        df = df.join(pd.DataFrame(param_array, columns = self.parameters.keys()))
        df.insert(df.shape[1], 'explainer' ,[explainer for _ in range(df.shape[0])])
        if save:
            df.to_csv("Results/"+explainer+"/"+explainer+"_vals_"+suffix+".csv")
        return df

        
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
alpha = [-1.0, -0.1, 1.0]
beta = [-1.0, -0.1, 1.0]
gamma = [0.2, 0.28, 0.37, 0.5, 0.65]
delta = [0.3, 1.0]
omega = [0, 0.3, 1.2]

def energy(x, v):
    return 0.5*v**2 + 0.5*alpha*x**2 +0.25*beta*x**4

parameters_now = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'omega':omega}
duffing = Duffing(parameters = parameters_now)
eom = duffing.eom
suffix = duffing.suffix

end_time = 100
duffing.generate(10000, samples = 100, end_time = end_time)
duffing.scale_features()
X = duffing.X_df[duffing.features]
y = duffing.X_df[duffing.labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
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
        
        
big_df.to_csv("Results/explainer_dataframe_full.csv")  

