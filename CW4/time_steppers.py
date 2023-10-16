import numpy as np
import matplotlib.pyplot as plt
import newton #(run newton_solver(...) as newton.newton_solver(...))

def theta_ode_solver(a,b,f,df,N,y0,theta):
    '''

    Parameters
    ----------
    a : float
        The start of the interval
    b : float
        The end of the interval, b > a
    f : function
        The function to be solved using theta-schemes method
    df : function
        The partial derivative of function f with respect to y
    N : positive integer
        The number of time-steps to take
    y0 : float
        The initial condition
    theta : float
        Belongs to[0,1], the choice of theta to be used in theta-schemes method

    Returns
    -------
    t: numpy.ndarray
        a numpy.ndarray of shape(N+1,) representing the points at which the function is approximated
    y: numpy.ndarray
        a numpy.ndarray of shape(N+1,) representing the approximations of y for a given theta   
        
    Examples
    --------
    >>> theta_ode_solver(0, 2, lambda t,y: y-t**2+1, lambda t,y: np.ones(np.shape(y)), 6, 0.5, 0.75) 
    '''
    
    t = np.zeros(N+1,) # Set t a numpy.ndarray of shape(N+1,)
    y = np.zeros(N+1,) # Set y a numpy.ndarray of shape(N+1,)
    
    h = (b-a)/N # Step size h
    
    t[0] = a # Initial condition
    y[0] = y0 # Initial condition
    
    for i in range(1,N+1):
        t[i] = a + i*h # Nodes t
        
        def fstar(x): 
            return y[i-1] + h*theta*f(t[i-1],y[i-1]) + h*(1-theta)*f(t[i],x) - x
        #For fixed t, generate the function of y to use Newton's method
        
        def dfstar(x):
            return h*(1-theta)*df(t[i],x) - 1
        #For fixed t, generate the function fstar partial derivative of y to use Newton's method
        
        y[i] = newton.newton_solver(fstar,dfstar,y0,100,1.0e-12) # Use Newton's method to solve theta-schemes equation
        
        y0 = y[i]
               
    return t, y






def runge_kutta(a,b,f,N,y0,m,method):
    '''
    Parameters
    ----------
    a : float
        The start of the interval
    b : float
        The end of the interval, b > a
    f : function
        The function to be solved using Runge-Kutta Schemes
    N : positive integer
        The number of time-steps to take
    y0 : float
        The initial condition
    m : positive integer
        Representing the given function f(t,y) is a m-vector function
    method : integer
        Takes the values of 1, 2 or 3.
        1: represents using the Forward Euler Method (RK1)
        2: represents using the Midpoint Method (RK2)
        3: represents using the Heun's Method of order three (RK3)
        
    Returns
    -------
    t : numpy.ndarray
        a numpy.ndarray of shape(N+1,) representing the points at which the function is approximated
    y : numpy.ndarray
        a numpy.ndarray of shape(m,N+1) contains the approximations to f given initial condition y0
        
    Examples
    --------
    >>> runge_kutta(0, 1, lambda t,y: np.array([y[0]-2*y[1]+t*y[2],-y[0]+y[1]+y[2]-t**2,t*y[1]-y[1]+t**3]), 4, np.array([1,0,1]), 3, 1) 
    '''
    
    t = np.zeros(N+1,) # Set t a numpy.ndarray of shape(N+1,)
    y = np.zeros((m,N+1)) # Set y a numpy.ndarray of shape(m,N+1)
    
    h = (b-a)/N # Step size h
    
    t[0] = a # Initial condition
    t0 = t[0]
    y[:,0] = y0
    
    for j in range(1,N+1):
        t[j] = a + j*h # Time points t
        
        if method == 1: # Forward Euler Method (RK1)
            y[:,j] = y0 + h*f(t0,y0)
            # Apply the formular of Forward Euler Method (RK1)
            y0 = y[:,j]
            t0 = t[j]
                   
        if method == 2: #  Midpoint Method (RK2)
            y[:,j] = y0 + h*f(t0+h/2,y0+h*f(t0,y0)/2)
            # Apply the formular of Midpoint Method (RK2)
            y0 = y[:,j]
            t0 = t[j]
        
        if method == 3: # Heun's Method of order three (RK3)
            y[:,j] = y0 + (h/4)*(f(t0,y0)+3*f(t0+2*h/3,y0+(2*h/3)*f(t0+h/3,y0+h*f(t0,y0)/3)))
            # Apply the formular of Heun's Method of order three (RK3)
            y0 = y[:,j]
            t0 = t[j]

    return t, y

