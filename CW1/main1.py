"""
MATH2019 CW1 main script

@author: Kris van der Zee (lecturer)
"""

import numpy as np
import matplotlib.pyplot as plt
import rootfinders as rf


#%% Question 2
 

# Initialise 
f = lambda x: x**3 + x**2 - 2*x - 2 
a = 1
b = 2
Nmax = 5

# Run bisection
p_array = rf.bisection(f,a,b,Nmax)

# Print output
print(p_array)


#%% Question 3

# Initialise 
f = lambda x: x**3 + x**2 - 2*x - 2 
c = 1/12
p0 = 1
Nmax = 10

# Run fixedpoint_iteration
p_array = rf.fixedpoint_iteration(f,c,p0,Nmax)

# Print output
print(p_array)

# Help
help(rf.fixedpoint_iteration)


#%% Question 4

# Initialise 
f = lambda x: x**2 - 2 
dfdx = lambda x: 2*x 
p0 = 1
Nmax = 8

# Run method
p_array = rf.newton_method(f,dfdx,p0,Nmax)

# Print output
print(p_array)


#%% Question 5

# Initialise 
f = lambda x: x**3 + x**2 - 2*x - 2 
dfdx = lambda x: 3*x**2 + 2*x - 2
c = 1/12
p0 = 1
p1 = 2
Nmax = 20
p_exact = np.sqrt(2)

# Run plot_convergence
fig = plt.figure()
out = rf.plot_convergence(p_exact,f,dfdx,c,p0,p1,Nmax,fig)
plt.show()


#%% Question 6(*)

# Initialise 
f = lambda x: x**2 - 2  
p0 = 1
p1 = 2
Nmax = 5

# Run method
p_array = rf.secant_method(f,p0,p1,Nmax)
print(p_array)

# Large Nmax to test zero division
Nmax = 12
p_array = rf.secant_method(f,p0,p1,Nmax)
print(p_array)
