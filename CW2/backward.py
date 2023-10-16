"""
MATH2019 CW2 backward module

@author: Richard Rankin
"""

import numpy as np

def backward_substitution(M,n):
    
    """
    Backward Substitution: Returns a numpy array of the 
    solution of a linear system (in echelon form).
    
    Parameters
    ----------
    M : numpy.ndarray
        Input matrix of size n x (n+1) (assumed to be in echelon form)
    n : integer
        Size of system
        
    Returns
    -------
    x : numpy.ndarray, shape (n,)
        Solution vector of augmented matrix M
    """
    
    # Initialise
    x=np.zeros([n,1])
    
    # First compute last one 
    x[n-1]=M[n-1,n]/M[n-1,n-1]
    
    # Loop backwards to compute others
    for i in range(n-2,-1,-1):
        x[i]=(M[i,n]-np.dot(M[i,i+1:n],x[i+1:n]))/M[i,i]
    
    # End
    return x
