"""
MATH2019 CW1 rootfinders module

@author: Zihang Yu
"""


import numpy as np
import matplotlib.pyplot as plt

def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    for i in range (Nmax):
        P = (a+b)/2
        if f(a)*f(P) >= 0:
            a = P
        else:
            b = P
        p_array[i] = P
    
    
    return p_array



def fixedpoint_iteration(f,c,p0,Nmax):
    
    """
    Fixed-point iteration method: Returns a numpy array of the 
    sequence of approximations obtained by the Fixed-point iteration method.
    
    Parameters
    ----------
    f : function
        Input function for which the root is to be found. 
    c: real number
        Used in the definition of g(x) = x - c*f(x).
    p0: real number
        Initial approximation to start the fixed-point interation.
    Nmax: integer
            Maximum number of interations.
    
    Returns
    -------
    p_array: numpy.ndarray, shape (Nmax,)
        Array of the approximations pn (n = 1,2,...).
    """
    
    p_array = np.zeros(Nmax)
    for i in range(Nmax):
        p1 = p0 - c*f(p0)
        p0 = p1

        p_array[i] = p0
    
    return p_array


def newton_method(f,dfdx,p0,Nmax):
    p_array = np.zeros(Nmax)
    for i in range(Nmax):
        p1 = p0 - f(p0)/dfdx(p0)
        p0 = p1
        p_array[i] = p0
        
    return p_array


def plot_convergence(p_exact,f,dfdx,c,p0,p1,Nmax,fig):
    
    p_array1 = bisection(f,p0,p1,Nmax)
    for i in range(1,Nmax+1):
        p_array1[i-1] = np.absolute(p_array1[i-1] - p_exact)
    
    p_array2 = fixedpoint_iteration(f,c,p0,Nmax)
    for i in range(1,Nmax+1):
        p_array2[i-1] = np.absolute(p_array2[i-1] - p_exact)
    
    p_array3 = newton_method(f,dfdx,p0,Nmax)
    for i in range(1,Nmax+1):
        p_array3[i-1] = np.absolute(p_array3[i-1] - p_exact)
    
    p_array4 = secant_method(f,p0,p1,Nmax)
    for i in range(1,Nmax+1):
        p_array4[i-1] = np.absolute(p_array4[i-1] - p_exact)
    
    n = np.arange(1,Nmax+1)
    list1 = []
    for i in range(Nmax+1):
        list1.append(i)
    plt.semilogy()
    plt.ylim(10**(-16),10**0)
    plt.xlim(0,Nmax+1)
    plt.scatter(n,p_array1,marker='s',label='bisection method')
    plt.scatter(n,p_array2,marker='^',label='fixedpoint iteration')
    plt.scatter(n,p_array3,marker='o',label="Newton's method")
    plt.scatter(n,p_array4,marker='+',label='secant method')
    plt.legend(loc="best",fontsize="x-small")
    plt.xticks(list1)
    plt.grid(True,linestyle=':')
    plt.show()
     
    return 1



def secant_method(f,p0,p1,Nmax):
    p_array = np.zeros(Nmax)
    for i in range(Nmax):
        if np.absolute(f(p1)-f(p0)) < 10**(-14):
            p2 = p1
            p1 = p0
            p_array[i] = p0
            
        else:
            p2 = p1 - f(p1)/((f(p1)-f(p0))/(p1-p0))
            p0 = p1
            p1 = p2
            p_array[i] = p0
    
    return p_array



