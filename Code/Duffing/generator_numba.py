import numpy as np

import numba
from numba import jit

class DataGenerator():
    def __init__(self, tol=1e-6, hmax=1e1, hmin=1e-16):
        self.tol=tol
        self.hmax=hmax 
        self.hmin=hmin
        self.function= None
    @numba.njit    
    def rkf(self, f, a, b, x0, tol, hmax, hmin ):

        # RK4(5) Fehlberg Coefficients
        a2  =   2.500000000000000e-01  #  1/4
        a3  =   3.750000000000000e-01  #  3/8
        a4  =   9.230769230769231e-01  #  12/13
        a5  =   1.000000000000000e+00  #  1
        a6  =   5.000000000000000e-01  #  1/2

        b21 =   2.500000000000000e-01  #  1/4
        b31 =   9.375000000000000e-02  #  3/32
        b32 =   2.812500000000000e-01  #  9/32
        b41 =   8.793809740555303e-01  #  1932/2197
        b42 =  -3.277196176604461e+00  # -7200/2197
        b43 =   3.320892125625853e+00  #  7296/2197
        b51 =   2.032407407407407e+00  #  439/216
        b52 =  -8.000000000000000e+00  # -8
        b53 =   7.173489278752436e+00  #  3680/513
        b54 =  -2.058966861598441e-01  # -845/4104
        b61 =  -2.962962962962963e-01  # -8/27
        b62 =   2.000000000000000e+00  #  2
        b63 =  -1.381676413255361e+00  # -3544/2565
        b64 =   4.529727095516569e-01  #  1859/4104
        b65 =  -2.750000000000000e-01  # -11/40

        r1  =   2.777777777777778e-03  #  1/360
        r3  =  -2.994152046783626e-02  # -128/4275
        r4  =  -2.919989367357789e-02  # -2197/75240
        r5  =   2.000000000000000e-02  #  1/50
        r6  =   3.636363636363636e-02  #  2/55

        c1  =   1.157407407407407e-01  #  25/216
        c3  =   5.489278752436647e-01  #  1408/2565
        c4  =   5.353313840155945e-01  #  2197/4104
        c5  =  -2.000000000000000e-01  # -1/5

        # Initialisation
        t = a
        x = np.array(x0)
        h = hmax

        T = np.array( [t] )
        X = np.array( [x] )

        # Update Loop
        while t < b:

            if t + h > b:
                h = b - t
            k1 = h * f( x, t )
            k2 = h * f( x + b21 * k1, t + a2 * h )
            k3 = h * f( x + b31 * k1 + b32 * k2, t + a3 * h )
            k4 = h * f( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h )
            k5 = h * f( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h )
            k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                        t + a6 * h )

            r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
            if len( np.shape( r ) ) > 0:
                r = max( r )
            if r <= tol:
                t = t + h
                x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
                T = np.append( T, t )
                X = np.append( X, [x], 0 )

            h = h * min( max( 0.84 * ( tol / r )**0.25, 0.1 ), 4.0 )

            if h > hmax:
                h = hmax
            elif h < hmin:
                raise RuntimeError("Error: Could not converge to the required tolerance %e with minimum stepsize  %e." % (tol,hmin))
                break

        return ( T, X )
    
    def generate(self,t, x0):
        self.t, self.u = self.rkf(f=self.function, a=0, b=t, x0=x0, tol=self.tol, hmax=self.hmax, hmin=self.hmin)
        
# class to generate Duffing values for given values and time t.
class DuffingGenerator(DataGenerator):
    def __init__(self, alpha=-1, beta=1, delta=0.3, gamma=0.37, omega=1.2):
        super().__init__()
        self.beta = beta
        self.alpha  = alpha
        self.delta = delta
        self.omega = omega
        self.gamma = gamma
        self.function=self.duff_eom
    @numba.njit        
    def duff_eom(self,u, t):
        x, dx = u
        ddx= self.gamma * np.cos(self.omega * t) - (self.delta * dx + self.alpha*x + self.beta * x**3)
        return np.array([dx,ddx])
    