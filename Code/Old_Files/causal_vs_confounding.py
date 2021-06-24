# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:10:45 2021

@author: wilke
"""
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers


## seed random library
rand.seed(123)
"""
Let's generate a dataset for a causal relationship: a(x) = w_1*x + w_2*x^2 + epsilon, b(a) = w_3*a^3 +epsilon_2;
And one for a set with a confounding variable: a(x) = w_1*x + w_2*x^2 + epsilon, b(x) = w_3*x^3 +epsilon_2;
"""

##CAUSAL
w_1 = 2.7
w_2 = 5.2

n = 50000

x_vec = np.random.rand(n, 2)
eps_vec = np.zeros(n)
for i in range(n):
    eps_vec[i] = rand.normalvariate(0,.001)
    

epsilon = rand.normalvariate(0,1)
a_vec = w_1*x_vec[:,0] + w_2*x_vec[:,1]**2 + eps_vec

def build_regressor():
    model = Sequential()
    model.add(layers.Dense(units = 2, input_dim = 2))
    model.add(layers.Dense(units = 1))
    model.compile(optimizer= 'adam', loss = 'mean_squared_error', metrics=['mae', 'accuracy'])
    return model


from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn = build_regressor, batch_size = 256, epochs = 128)


results = regressor.fit(x_vec[:int(9*n/10),:], a_vec[:int(9*n/10)])

predictions = regressor.predict(x_vec[int(9*n/10):,:])


plt.plot(x_vec[int(9*n/10):,1], predictions, 'x')
plt.plot(x_vec[int(9*n/10):,1], a_vec[int(9*n/10):], 'o')



# plt.plot(x_vec[:,0], a_vec, 'x')

