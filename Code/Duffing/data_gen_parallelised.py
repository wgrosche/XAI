# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:03:54 2021

@author: wilke
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import multiprocessing as mp


#import progressbar as pb

# class DataGenerator():
#     def __init__(self, gamma = 0.37, x_max = 2, x_min = -2, v_max = 1, v_min = -1):
#         self.x_max = x_max
#         self.x_min = x_min
#         self.v_max = v_max
#         self.v_min = v_min
#         self.t_range = np.linspace(0, 50, 500, endpoint=False)
#         self.gamma = gamma
#         #Initialise the output arrays
        
#     def eom(self, u, t):
#         alpha=-1
#         beta=1
#         delta=0.3
#         gamma=self.gamma
#         omega=1.2
#         x, dx = u[0], u[1]
#         ddx= gamma * np.cos(omega * t) - (delta * dx + alpha*x + beta * x**3)
#         return [dx,ddx]
       
#     def sample_single_traj(self, i):
#         x0 = (self.x_max - self.x_min) * np.random.random_sample() + self.x_min
#         v0 = (self.v_max - self.v_min) * np.random.random_sample() + self.v_min
#         #Generate a trajectory
#         trajectory = odeint(self.eom, [x0,v0], self.t_range)
#         #Sample a random point along the trajectory
#         t2_index = np.random.randint(0, len(self.t_range))
#         Xi = [x0,v0,self.t_range[t2_index]]
#         yi = trajectory[t2_index,:] 
#         return Xi, yi
      
#     def generate(self, num_samples):
#         print("test")
#         self.X = np.empty((num_samples, 3))
#         self.y = np.empty((num_samples, 2))
#         pool = mp.Pool(mp.cpu_count())
#         res = pool.map(self.sample_single_traj, [sample for sample in range(num_samples)])
#         pool.close()
#         return res

"""
Duffing Oscillator Equation of motion, given x0,v0 and t it returns x_dot, v_dot
"""



"""
Samples randomly from x0 in [-2,2], v0 in [-1,1]
"""

def eom(self, u, t):
    alpha=-1
    beta=1
    delta=0.3
    gamma=0.37
    omega=1.2
    x, dx = u[0], u[1]
    ddx= gamma * np.cos(omega * t) - (delta * dx + alpha*x + beta * x**3)
    return [dx,ddx]

def gen_sample(sample):
    x0 = (x_max - x_min) * np.random.random_sample() + x_min
    v0 = (v_max - v_min) * np.random.random_sample() + v_min
    trajectory = odeint(eom, [x0,v0], t_range)
    t2_index = np.random.randint(0, len(t_range))
    X[sample,:] = [x0,v0,t_range[t2_index]]
    y[sample,:] = trajectory[t2_index,:]
    
if __name__ == "__main__":
    #mp.freeze_support()
    global x_max, x_min, v_max, v_min 
    global X, y
    global t_range
    pool = mp.Pool(mp.cpu_count())
    num_samples = 1e8
    X = np.empty((num_samples, 3))
    y = np.empty((num_samples, 2))
    #Define bounds of the sampling
    x_max = 2
    x_min = -2
    v_max = 1
    v_min = -1
    #Define the t_range to draw from
    t_range = np.linspace(0, 50, 500, endpoint=False)
    pool.map(gen_sample, [sample for sample in range(num_samples)])
    pool.close()
    pd.DataFrame(X, columns=['x0','v0','t']).to_csv("X_train.csv")
    pd.DataFrame(y, columns=['xt','vt']).to_csv("y_train.csv")
#Generate num_samples samples
#with pb.ProgressBar(max_value=num_samples) as bar:
    
# res = pool.map(self.sample_single_traj, [sample for sample in range(num_samples)])
#         pool.close()
# for i in range(num_samples):
#     #Generate random starting positions
#     x0 = (x_max - x_min) * np.random.random_sample() + x_min
#     v0 = (v_max - v_min) * np.random.random_sample() + v_min
#     #Generate a trajectory
#     func = lambda u,t: eom(u, t, gamma)
#     trajectory = odeint(func, [x0,v0], t_range)
#     #Sample a random point along the trajectory
#     t2_index = np.random.randint(0, len(t_range))
#     X[i,:] = [x0,v0,t_range[t2_index]]
#     y[i,:] = trajectory[t2_index,:]
#         #bar.update(i)
        
#     return X, y

# def sample_single_traj():
#     x0 = (x_max - x_min) * np.random.random_sample() + x_min
#     v0 = (v_max - v_min) * np.random.random_sample() + v_min
#     #Generate a trajectory
#     func = lambda u,t: eom(u, t, gamma)
#     trajectory = odeint(func, [x0,v0], t_range)
#     #Sample a random point along the trajectory
#     t2_index = np.random.randint(0, len(t_range))
#     X[i,:] = [x0,v0,t_range[t2_index]]
#     y[i,:] = trajectory[t2_index,:]
    
# def main():
#     gen = DataGenerator()
#     res = gen.generate(int(10e1))
#     print(res)
#     #Generate the data
#     # X_train, y_train = gen.generate(int(10e1))
#     # X_test, y_test = gen.generate(int(10e1))
    
#     #Save the generated data in pd dataframes
#     # pd.DataFrame(X_train, columns=['x0','v0','t']).to_csv("X_train.csv")
#     # pd.DataFrame(X_train, columns=['x0','v0','t']).to_csv("y_train.csv")
#     # pd.DataFrame(X_train, columns=['x0','v0','t']).to_csv("X_test.csv")
#     # pd.DataFrame(X_train, columns=['x0','v0','t']).to_csv("y_test.csv")


# if __name__ == "__main__":
#     #mp.freeze_support()
#     main()