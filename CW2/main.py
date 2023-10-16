"""
MATH2019 CW2 main script

@author: Dr Richard Rankin
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Question 1

import systemsolvers as ss

# Initialise 
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
b = np.array([[7],[6],[4]])
n = 3

# Run no_pivoting
M = ss.no_pivoting(A,b,n,1)
# Print output
print(M)

# Run no_pivoting
M = ss.no_pivoting(A,b,n,2)
# Print output
print(M)

# Solve Ax=b
x = ss.no_pivoting_solve(A,b,n)
# Print output
print(x)

# Help
help(ss.no_pivoting_solve)


#%% Question 2


import systemsolvers as ss
A = np.array([[1,2,-1,1],[14,-1,12,-1],[0.0,1,-10,1],[15,3,-13,-1]])
b = np.array([[1],[3],[2],[4]])
n = 4
c = 2
M = ss.partial_pivoting(A,b,n,c)
# Initialise
# A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
# b = np.array([[7],[6],[4]])
# n = 3

# # Run find_max
# p = ss.find_max(np.hstack((A,b)),n,0)
# # Print output
# print(p)

# # Run partial_pivoting
# M = ss.partial_pivoting(A,b,n,1)
# Print output
print(M)

# Run find_max
p = ss.find_max(M,n,1)
# Print output
print(p)

# Run partial_pivoting
M = ss.partial_pivoting(A,b,n,2)
# Print output
print(M)

# Solve Ax=b
x = ss.partial_pivoting_solve(A,b,n)
# Print output
print(x)


#%% Question 3

import systemsolvers as ss

# Initialise
A = np.array([[1,1,0.0],[2,1,-1],[0,-1,-1]])
n = 3

# Run Doolittle
L, U = ss.Doolittle(A,n)
# Print output
print(L)
print(U)


#%% Question 4

import systemsolvers as ss

# Initialise
A = np.array([[4,-1,0],[-1,8,-1],[0,-1,4]])
b = np.array([[48],[12],[24]])
n = 3
x0 = np.array([[1.0],[1],[1]])
tol=1e-2

# Run Gauss_Seidel
x = ss.Gauss_Seidel(A,b,n,x0,tol,3)
# Print output
print(x)

# Run Gauss_Seidel
x = ss.Gauss_Seidel(A,b,n,x0,tol,4)
# Print output
print(x)
