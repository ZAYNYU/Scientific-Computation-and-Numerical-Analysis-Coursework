import numpy as np

def newton_solver(f,df,x0,Nmax,tol):
  """
  Use the Newton-Raphson method to find a root of f(x) given
  an initial guess x0

  Parameters
  ----------
  f : function of one variable
        function to find the root of
  df : function of one variable
        derivate of f
  x0 : float
        Initial guess
  Nmax : int
        Maximum number of iterations to perform
  tol : float
        Tolerance at which the approximate root will accepted

  Returns
  -------
    float
        The approximation to the root.

  Examples
  --------
  >>> newton_solver(lambda x: x**2+1,lambda x: 2*x,0.5,100,1.0e-12)
  """

  #Initialise
  xstar = x0
  xold = xstar

  #Perform the Newton iteration
  for k in range(Nmax):
    xstar = xstar-f(xstar)/df(xstar)

    if np.abs(xstar-xold) < tol:
      return xstar

    xold = xstar

  print("Warning, Newton solver has not converged")
  return xstar

