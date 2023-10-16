"""
MATH2019 CW4 polynomial_interpolation module
"""

### Load other modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

### Comment out incomplete functions before testing ###

#%%
def richardson(f,x0,h,k):
    '''
    Returns the approximation of the first derivative of given function based on formular (3),
    given width and level using Richardson extrapolation. 
    
       
    Parameters
    ----------
    f : function
        the function to be approximated
    x0 : float
        the point at which the function will be approximated
    h : float
        h > 0, the width of the (three-point) central difference approximation
    k: integer
        k > 0, the level of Richardson extrapolation
    
    Returns
    -------
    deriv_approx : float
                   the approximation N_k(h) to f'(x_0) of given single point x_0, width h and level k
                   
    Examples
    --------
    >>> richardson(lambda x: np.sin(10*x),0.5,0.1,3)
    '''
    N1 = lambda t: (f(x0+t)-f(x0-t))/(2*t) # N1 is the function of h which different h values will be evaluate
    if k == 1:
        deriv_approx = N1(h)
        
    else:
        a = np.zeros((k,)) # Zero array of shape((k,)) to store the coefficient of h values
        for i in range(k):
            a[i] = 2**i # Calculate the coefficient of h values
        hvals = h/a # Find h values need to be evaluate by the function N1
        
        N1_1 = N1(hvals) # Find the first approximation
        
        for j in range(1,k):
            
            a = 1/(1-4**j) # Find coefficient a by the given formular
            b = 4**j/(4**j-1) # Find coefficient b by the given formular
            Nk = []
            for u in range(len(N1_1)-1): # Times we need to calculate for Nk
                Nk.append(a*N1_1[u] + b*N1_1[u+1]) # Use function (3)
            N1_1 = Nk # Replace N1_1 by Nk
        
        deriv_approx = Nk[0] # The first and also the only terms left in Nk
        
    return deriv_approx

#%%
def richardson_errors(f,f_deriv,x0,n,h_vals,k_max):
    '''
    Comments and Explaination
    -------
    Applied the function richardson_errors when x0 = 0, k_max = 4 and h_vals = np.logspace(-5,1,20)
    (a) f(x) = sin(10x)
    (b) f(x) = -x^2 if x<=0, f(x) = 0 if x>0
    
 1. In general, the results of the two functions on the plots show that both of the errors of the Richardson method get smaller as h decreases.
    This is because Nk(h) tends to f'(x0) as h tends to 0, therefore, Error = |f'(x0) - Nk(h)| tends to 0.
    When h is the same, the bigger level k is, the smaller the error. This is because the derivation of the Richardson extrapolation formula shows that
    the order of error increases as k increases.
    
    2. For function (a), the plot shows strange behaviour(fractures and vertical lines) when h is small enough and k is bigger than 1.
    While the output result indicates that this is because at these points the errors are equal to zero, which means that we get the exact value,
    but as h decreases there are some errors that are not zero, this might because round-off error.
    Another notable feature is that for function(a) the error converges faster than function(b).
    We can verify this by ploting a straight line with gradient 1 of h_vals(plt.loglog(h_vals,h_vals)).
    This indicates that for function(a) the error has higher-order convergence than function(b), which converges of O(h)(see below).
    
    3. For function (b), the rate of error convergence is not as fast as function(a). Furthermore, for different values of levels,
    the graphs of h values and errors are parallel straight lines. This indicates that Error(h) is a linear function of h, and the error converges of O(h).
    The reason is that, set the initial value into function(b), we get N1(h) = h/2, Error1(h) = |N1(h)|, are linear functions of h.
    Since Nk(h) is linear combination of N1(h), therefore Errork(h) = |Nk(h)| is also linear. 
    Taking log on x and y when doing the plot doesn't change the linear property.
    '''
    error_matrix = np.zeros((k_max,n)) # Set the error matrix to be a numpy.ndarray of shape (k_max,n)
        
    for i in range(1,k_max+1):
        for j in range(n):
            E = abs(f_deriv(x0) - richardson(f,x0,h_vals[j],i)) # Calculate the errors of level i
            error_matrix[i-1][j] = E # The (i-1)j-th entry of error matrix
        
    fig = plt.figure() #This line is required (once) before any other plotting commands
    plt.title("Richardson Errors") # Show title
    plt.xlabel('$h$') # Show xlabel
    plt.ylabel('Errors') # Show ylabel
    
    for u in range(k_max):
        Ek = error_matrix[u] # Errors for each level
        plt.loglog(h_vals,Ek,label =f'level {u+1}', marker = '.') # Plot the errors against h for each level
    plt.legend() # Show legend
    # plt.loglog(h_vals,h_vals) # A straight line with gradient 1 of h_vals for can be used to compare order of convergence
    return error_matrix, fig
