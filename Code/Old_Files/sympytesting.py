#sympy testing

from sympy import series, Symbol
from sympy.functions import sin, cos, exp

x = Symbol('x')

# def function(x):
#     return -gamma*dx/dt + 2*a*x - 4*b*x^3 + F_0*cos(omega*t)

def taylor(function, x0, n):
    """
    Parameter "function" is our function which we want to approximate
    "x0" is the point where to approximate
    "n" is the order of approximation
    """
    return function.series(x,x0,n).removeO()

print('sin(x) =', taylor(sin(x), 0, 4))