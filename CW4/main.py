#Import regular modules
import numpy as np
import matplotlib.pyplot as plt

#Import student modules
import numerical_differentiation as nd
import time_steppers as ts

#%% Question 1
print('Q1 test output:')

f = lambda x: np.sin(10*x)
h = 0.1
x0 = 0.5
deriv_approx1 = nd.richardson(f,x0,h,2)
deriv_approx2 = nd.richardson(f,x0,h,3)


print("deriv_approx1 = "+str(deriv_approx1))
print("deriv_approx2 = "+str(deriv_approx2))


#%% Question 2
print('\n')
print('Q2 test output:')

f = lambda x: np.sin(10*x)
f_deriv = lambda x: 10.0*np.cos(10*x)
n = 4
h_vals = [0.5, 0.1, 0.05, 0.01]
k_max = 3
x_0 = 0.0
error_matrix, fig = nd.richardson_errors(f,f_deriv,x_0,n,h_vals,k_max)


print("error_matrix = ")
print(error_matrix)
print(help(nd.richardson_errors))

#%% Question 3
print('\n')
print('Q3 test output:')

a = 0; b = 2
N = 6
theta = 0.75
f = lambda t,y: y-t**2+1
df = lambda t,y: np.ones(np.shape(y))
y0 = 0.5
t,y = ts.theta_ode_solver(a,b,f,df,N,y0,theta)


print("t = "+str(t))
print("y = "+str(y))

#%% Question 4
print("\n")
print('Q4 test output:')

a = 0; b = 1
N = 4
m = 3
f = lambda t,y: np.array([y[0]-2*y[1]+t*y[2],\
                          -y[0]+y[1]+y[2]-t**2,t*y[1]-y[1]+t**3])
y0 = np.array([1,0,1])

#Test 1
method = 1
t,y = ts.runge_kutta(a,b,f,N,y0,m,method)

print("Method1:")
print("t = "+str(t))
print("y = ")
print(y)

#Test 2
method = 2
t,y = ts.runge_kutta(a,b,f,N,y0,m,method)

print("\n")
print("Method2:")
print("t = "+str(t))
print("y = ")
print(y)

#Test 3
method = 3
t,y = ts.runge_kutta(a,b,f,N,y0,m,method)
print("\n")
print("Method3:")
print("t = "+str(t))
print("y = ")
print(y)