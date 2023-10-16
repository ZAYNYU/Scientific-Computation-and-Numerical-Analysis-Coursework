"""
MATH2019 CW2 systemsolvers module

@author: Zihang Yu 20217334
"""

import numpy as np
import matplotlib.pyplot as plt
import backward as bw

def no_pivoting(A,b,n,c):
    
    """
    Returns the augmented matrix M arrived at by starting from the augmented
    matrix [A b] and performing forward elimination without row interchanges
    until all of the entries below the main diagonal in the first c columns
    are 0.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        positive integer.
    c : integer
        positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        2-D array representing the matrix M.
    """
    
    # Create the initial augmented matrix
    M=np.hstack((A,b))
    
    for i in range(0,c):
        for j in range(i+1,n):
            g = (M[j,i])/(M[i,i])
            M[j,i] = 0
            for k in range(i+1,n+1):
                M[j,k] = (M[j,k])-g*(M[i,k])
        
    
    return M



def no_pivoting_solve(A,b,n):
    
    """
    Returns the solution vector x of the linear system Ax = b 
    by Gaussian elimination without using pivoting strategies.
    
    Parameters
    ----------
    A : numpy.ndarray
        array representing the matrix A in the linear system Ax=b.
    b : numpy.ndarray
        array representing the vector b in the linear system Ax=b.
    n : integer
        positive integer.

    Returns
    -------
    x : numpy.ndarray, shape (n,1)
        solution vector of the linear system Ax = b.
    """
    
    M = no_pivoting(A, b, n, n-1)
    
    x = bw.backward_substitution(M,n)


    return x




def find_max(M,n,i):
    m = M[i,i]
    p = i
    for j in range(i+1,n):
        if np.abs(M[j,i]) > np.abs(m):
            m = M[j,i]
            p = j
    return p
    
    
    
    
def partial_pivoting(A,b,n,c):
    M = np.hstack((A,b))
    
    for i in range(0,c):
        p = find_max(M, n, i)
        M[[i,p],:] = M[[p,i],:]
    for j in range(0,c):
        for k in range(j+1,n):
            g = (M[k,j])/(M[j,j])
            M[k,j] = 0
            for l in range(j+1,n+1):
                M[k,l] = (M[k,l])-g*(M[j,l])
    
    return M
    


def partial_pivoting_solve(A,b,n):
    
    M = partial_pivoting(A,b,n,n-1)
    
    x = bw.backward_substitution(M,n)

    return x
        


def Doolittle(A,n):
    L = np.identity(n)
    U = A
    for i in range(0,n):
        for j in range(i+1,n):
            m = (A[j,i])/(A[i,i])
            L[j,i] = np.abs(m)
            U[j,i] = 0
            for k in range(i+1,n):
                U[j,k] = (A[j,k])-m*(A[i,k])
    
        
    return L , U



def Gauss_Seidel(A,b,n,x0,tol,maxits):
    
    D = np.diag(np.diag(A))
    L = np.tril(-A,-1)
    U = np.triu(-A,1)
    
    T = (np.linalg.inv(D-L))@U
    C = (np.linalg.inv(D-L))@b
    
    for i in range(1,maxits+1):
        x1 = T@x0 + C
        x0 = x1
        if np.max(np.abs(b-np.matmul(A,x0))) < tol:
            x = x0
            break
        else:
            x = ("Desired tolerance not reached after maxits iteration have been performed.")


    return x






