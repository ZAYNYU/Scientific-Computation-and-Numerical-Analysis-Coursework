"""
MATH2019 CW3 polynomial_interpolation module
"""

### Load other modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

### Comment out incomplete functions before testing ###

#%%
def lagrange_poly(p,xhat,n,x,tol):

    """
    Returns the matrix of Lagrange polynomials based on points in set xhat and x, 
    by using the method of Lagrange interpolating polynomials.
    
    Returns 0 or 1. if points in set xhat are distinct return 0, otherwise return 1.
    
    Parameters
    ----------
    p : integer
        positive integer
    n : integer
        positive integer
    xhat : numpy.ndarray of shape (1,p+1)
           the set of p+1 distinct nodal points
    x : numpy.ndarray of shape (1,n)
        the set of n evaluation points
    tol : float
         tolerance of error to decide if two numbers can be considered equal
    
    Returns
    -------    
    lagrange_matrix : numpy.ndarray of shape (p+1,n)
                      the matrix of Lagrange polynomials
    
    error_flag : 0 or 1
                 if points in set xhat are distinct error_flag = 0, otherwise error_flag = 1   
    
    """  
    lagrange_matrix = np.zeros((p+1,n)) # Zero matirx of shape(p+1,n)
    error_flag = 0 # Assume the elements in xhat are distinct at the start
    for i in range(p+1): #Calculate Li(x),that is to calculate the i-th row of lagrange_matrix
        L = 1
        for j in range(p+1):
            if i == j:
                continue
            if np.abs((xhat[i])-(xhat[j])) < tol: # The elements in xhat are not distinct
                error_flag = 1
                break # Stop the calculation to avoid divide by zero
            else:    
                L = L * ((x-xhat[j])/(xhat[i]-xhat[j])) # Use the method of Lagrange interpolating polynomials
        
        lagrange_matrix[i:] = L
    
    if error_flag == 1: 
        lagrange_matrix = None # If the elements in xhat are not distinct, return None

    return lagrange_matrix, error_flag

#%%
def uniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    """
    Returns the p-th order polynomial interpolant of function f at a set of points x, 
    by applying the method of Lagrange interpolating polynomials when a uniform set of interpolating nodes is used.
    
    Returns the figure of the function f and the interpolant or None.
    
    Parameters
    ----------
    a : real number
        the start of interval
    b : real number
        the end of interval, b > a
    p : integer
        positive integer
    n : integer
        positive integer
    x : numpy.ndarray of shape (1,n)
        the set of n evaluation points
    f : function
        the function to be interpolated
    produce_fig : True or False
                  Plot when input is True and None if input is False
    
    Returns
    -------    
    interpolant : numpy.ndarray of shape (n,)
                  representing the p-th order polynomial interpolant of function f
    
    fig : matplotlib.figure.Figure when input produce_fig is True
          representing the function of f and the interpolant
          None if produce_fig is False
          
    """    
    xhat = np.linspace(a,b,p+1) # Set xhat into lagrange_poly from Q1
    L = (lagrange_poly(p,xhat,n,x,1.0e-10))[0] # Call lagrange_poly from Q1
    interpolant = np.zeros((n,)) # Zero array of shape (n,)
    
    interpolant = f(xhat) @ L # Use the formula of Lagrange interpolating polynomials
  
    if produce_fig == True: # Plot figure
        fig = plt.figure()
        plt.title("function $f$ and the interpolant $P_p(x)$ using uniform nodal points") # Set title
        plt.plot(x,f(x),label = '$f(x)$', marker = '.') # Plot the function f
        plt.plot(x,interpolant,label ='$P_p(x)$', marker = '.') # Plot the interpolant p(x)
        plt.legend() # Show legend
        plt.xlabel('$x$') # Show xlabel
        plt.ylabel('$f(x),P_p(x)$') # Show ylabel
    elif produce_fig == False: # No plot
        fig = None
    
    return interpolant, fig

#%%
def nonuniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    """
    Returns the p-th order polynomial interpolant of function f at a set of points x, 
    by applying the method of Lagrange interpolating polynomials when a non-uniform set of interpolating nodes is used.
    
    Returns the figure of the function f and the interpolant or None.
    
    Parameters
    ----------
    a : real number
        the start of interval
    b : real number
        the end of interval, b > a
    p : integer
        positive integer
    n : integer
        positive integer
    x : numpy.ndarray of shape (1,n)
        the set of n evaluation points
    f : function
        the function to be interpolated
    produce_fig : True or False
                  Plot when input is True and None if input is False
    
    Returns
    -------    
    nu_interpolant : numpy.ndarray of shape (n,)
                  representing the p-th order polynomial interpolant of function f 
                  evaluate at nonuniformly spaced nodal interpolation points
    
    fig : matplotlib.figure.Figure when input produce_fig is True
          representing the function of f and the interpolant
          None if produce_fig is False
          
    """
    xhat = np.zeros((p+1,)) # Set xhat to be zero array of shape (p+1,)
    for i in range(p+1):
        xhat[i] = a + ((np.cos(((2*i+1)/(2*(p+1)))*(np.pi))+1)/2)*(b-a)
        # Map the set of nonuniform nodal points on the interval (-1,1) to (a,b) hence get xhat
    
    L = (lagrange_poly(p,xhat,n,x,1.0e-10))[0] # Call lagrange_poly from Q1    
    nu_interpolant = np.zeros((n,)) # Zero array of shape (n,)
    
    nu_interpolant = f(xhat) @ L # Use the formula of Lagrange interpolating polynomials

    if produce_fig == True: # Plot figure
        fig = plt.figure()
        plt.title("function $f$ and the interpolant $P_p(x)$ using nonuniform nodal points") # Set title
        plt.plot(x,f(x),label = '$f(x)$', marker = '.') # Plot the function f
        plt.plot(x,nu_interpolant,label ='$P_p(x)$', marker = '.') # Plot the interpolant p(x)
        plt.legend() # Show legend
        plt.xlabel('$x$') # Show xlabel
        plt.ylabel('$f(x),P_p(x)$') # Show ylabel
    
    elif produce_fig == False: # No plot
        fig = None    
    
    return nu_interpolant, fig

#%%
def compute_errors(a,b,n,P,f):
    
    """
    Returns the errors of Lagrange Interpolation when uniform and non-uniform set of interpolating nodes is used
    
    Returns the figure of uniform and non-uniform error of Lagrange Interpolation.
    
    Parameters
    ----------
    a : real number
        the start of interval
    b : real number
        the end of interval, b > a
    n : integer
        positive integer
    P : numpy.ndarray
        a range of polynomial degrees
    f : function
        the function to be interpolated
    
    Returns
    -------    
    error_matrix : numpy.ndarray of shape (2,n)
                   errors of Lagrange Interpolation
                   the zeroth row represents the errors when a uniform set of interpolating nodes is used
                   the first row represents the errors when a non-uniform set of interpolating nodes is used
    fig : matplotlib.figure.Figure
          representing the uniform and non-uniform error against P on the same axes
 
    Comments and Explanation
    -----------------------
    Applied the above function when P = {1,2,3,...,40} and
    (a) f(x) = cos(2pix), [a,b] = [-1,1]
    (b) f(x) = cos(2pix) + (1/100)cos(10pix), [a,b] = [-1,1]
    
    The results of the two functions on the plots show that when n is small, both the uniform and non-uniform interpolating are equally good.
    As n increases, the picture shows that the error using uniform interpolating of both functions increases significantly.
    Especially for the second function for which the error even increases to around 1000.
    While the errors of both functions using non-uniform interpolating still decreases as n increases.
    
    The reason for error increase significantly as n gets bigger is because when n increase, the n-th order derivatives of this particular function increases exponentially,
    Especially for the second function because differentiate the term (1/100)cos(10pix) every time will multiply by 10pi.
    Therefore, from Theorem 3.3 in the lecture slides the formula of error for polynomial interpolation shows that the error also increases significantly. This is known as Runge's phenomenon. 
    The non-uniform interpolating method does not have the same problem as using uniform interpolating because the nodal interpolation points are nonuniformly
    spaced over the interval [a,b] and linearly scaled and shifted version of a function in Q3.
    The function which generates non-uniform points in Q3 works because they are clustered points sets that make the nodal points be of 
    asymptotic density on the interval [a,b] when n is large. They are known as Chebyshev points of the first kind from the family of Chebyshev points.
    
    """
      
    error_matrix = np.zeros((2,n)) # Set error_matrix to be zero matrix of shape (2,n)
    
    x = np.linspace(a,b,2000) # Evaluate the error for 2000 equally spaced points over [a,b]
    
    errors = []   # empty list for errors of uniform interpolating
    nu_errors = [] # empty list for errors of non-uniform interpolating
    
    for p in P:
        interpolants = (uniform_poly_interpolation(a,b,p,2000,x,f,False))[0] # Call function uniform_poly_interpolation
        nu_interpolants = (nonuniform_poly_interpolation(a,b,p,2000,x,f,False))[0] # Call function nonuniform_poly_interpolation
        
        error = np.abs(interpolants - f(x)) # Compute the error for uniform interpolating
        nu_error = np.abs(nu_interpolants - f(x)) # Compute the error for non-uniform interpolating
        
        
        errors.append(max(error)) # Find the max of errors of uniform interpolating
        nu_errors.append(max(nu_error)) # Find the max of errors of non-uniform interpolating 
        

    error_matrix[0,] = errors # Zeroth row contians the errors when using uniform set of interpolating nodes
    error_matrix[1,] = nu_errors # First row contians the errors when using non-uniform set of interpolating nodes
        
    
    fig = plt.figure()
    plt.title("Lagrange Interpolation Errors") # Set title
    plt.semilogy(P,error_matrix[0,],label = 'errors of uniform intepolating', marker = '.') # Plot the errors of uniform interpolating
    plt.semilogy(P,error_matrix[1,],label ='errors of non-uniform intepolating', marker = '.') # Plot the errors of non-uniform interpolating 
    plt.legend() # Show legend
    plt.xlabel('$P_n$') # Show xlabel
    plt.ylabel('$error$') # Show ylabel    
        
    return error_matrix, fig

#%%
def piecewise_interpolation(a,b,p,m,n,x,f,produce_fig):

    """
    Returns the p-th order polynomial interpolant of function f at a set of points x, 
    by using the method of Piecewise Polynomial Interpolation.
    
    Returns the figure of the function f and the interpolant or None.
    
    Parameters
    ----------
    a : real number
        the start of interval
    b : real number
        the end of interval, b > a
    p : integer
        positive integer    
    m : integer
        representing m uniform subintervals
    n : integer
        positive integer
    x : numpy.ndarray of shape (1,n)
        the set of n evaluation points
    f : function
        the function to be interpolated
    produce_fig : True or False
                  Plot when input is True and None if input is False
    
    Returns
    -------    
    p_interpolant : numpy.ndarray of shape (n,)
                  representing the p-th order piecewise polynomial interpolant of function f
    
    fig : matplotlib.figure.Figure when input produce_fig is True
          representing the function of f and the piecewise interpolant
          None if produce_fig is False
    """    
    
    
    xhat = np.zeros((p+1,)) # Set xhat to be zero array of shape (p+1,)
    for i in range(p+1):
        xhat[i] = np.cos(((2*i+1)/(2*(p+1)))*(np.pi)) # Calculate xhat as in Q3
    r1 = (xhat[0]-(-1))/(xhat[p]-xhat[0]) # The ratio to be used in rescaling the strat point
    r2 = (1-xhat[p])/(xhat[p]-xhat[0]) # The ratio to be used in rescaling the end point
    
    p_interpolant = np.zeros((n,)) # Set p_interpolant to be zero array of shape (n,)
    
    for i in range(m):
        p_a = a+(b-a)*(i/m) # The start of subintervals for piecewise interpolation
        p_b = a+(b-a)*((i+1)/m) # The end of subintervals for piecewise interpolation
        for j in range(n):             
            if p_a <= x[j] and x[j] <= p_b: # Decide which subinterval is x in
                p_aa = p_a - (p_b-p_a)*r1 # Rescale the start point of subinterval to make sure the interpolation is continuous
                p_bb = p_b + (p_b-p_a)*r2 # Rescale the end point of subinterval to make sure the interpolation is continuous
                p_interpolant[j] = (nonuniform_poly_interpolation(p_aa,p_bb,p,n,x,f,False)[0])[j] # Call function nonuniform_poly_interpolation
            
    if produce_fig == True: # Plot figure
        fig = plt.figure()
        plt.title("function $f$ and the piecewise interpolant $S^m_p(x)$") # Set title
        plt.plot(x,f(x),label = '$f(x)$', marker = '.') # Plot the function f
        plt.plot(x,p_interpolant,label ='$S^m_p(x)$', marker = '.') # Plot the piecewise interpolant S
        plt.legend() # Show legend
        plt.xlabel('$x$') # Show xlabel
        plt.ylabel('$f(x),S^m_p(x)$') # Show ylabel
    elif produce_fig == False: # No plot
        fig = None
    

    return p_interpolant, fig