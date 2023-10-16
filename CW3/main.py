import numpy as np
import matplotlib.pyplot as plt
import polynomial_interpolation as p_int

#%% Question 1
# Initialise
p = 3
xhat = np.linspace(0.5,1.5,p+1)
n = 7
x = np.linspace(0,2,n)
tol = 1.0e-10
#Run the function
lagrange_matrix, error_flag = p_int.lagrange_poly(p,xhat,n,x,tol)

print('Q1 test output:')
print(lagrange_matrix)
print(error_flag)

#%% Question 2
# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
interpolant, fig = p_int.uniform_poly_interpolation(a,b,p,n,x,f,False)

print('\n')
print('Q2 test output:')
print(interpolant)

#%% Question 3
# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
nu_interpolant, fig = p_int.nonuniform_poly_interpolation(a,b,p,n,x,f,False)

print('\n')
print('Q3 test output:')
print(nu_interpolant)

#%% Question 4
# Initialise
n = 5
P = np.arange(1,n+1)
a = -1
b = 1
f = lambda x: 1/(x+2)
#Run the function
error_matrix, fig = p_int.compute_errors(a,b,n,P,f)
print('\n')
print('Q4 test output:')
print(error_matrix)

#%% Question 5
#Initialise
a = -1
b = 1
n = 7
m = 5
x = np.linspace(-0.9,0.9,n)
f = lambda x: 1/(x+2)

#Test 1
p = 2
#Run the function
p_interpolant1, fig = p_int.piecewise_interpolation(a,b,p,m,n,x,f,False)

#Test 2
p = 10
#Run the function
p_interpolant2, fig = p_int.piecewise_interpolation(a,b,p,m,n,x,f,False)

print('\n')
print('Q5 test output:')
print(p_interpolant1)
print(p_interpolant2)
