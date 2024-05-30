"""
Genetic Algorithm Optimization of RL Predicted Curve - GA Class and Evaluation Functions
-----------------------------------------------------------------------------------------
    This file contains the GA Class and the Related Evaluation Functions
    
    Microstrip Arrangement
    ----------------------
    a: half width microstrip arrangement
    d: half width microstrip
    b: height of arrangement
    
    Note
    -----------------------
    values at x = 0 , x = d , and x = a are fixed and the input curve must not have value between x = 0 and x = d. 
    Otherwise it affects the size of the curve and thereby affects the program (within the functions to calculate small_v and energy).
"""

# import libraries
#-------------------------------
import numpy as np
from scipy.special import logsumexp
from pymoo.core.problem import Problem
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.optimize import minimize
# from pymoo.termination import get_termination

#----------------------------------------------------------------------------------------------------------------------------------
# functions for attribute calculation
#----------------------------------------------------------------------------------------------------------------------------------

def logsinh(x):
    """
    Auxiliary function
    Computes the natural logarithm of the sinh function, bypassing overflows that would happen if the sinh would be directly computed.
    """
    ones = np.ones_like(x)
    return logsumexp([x, -x], b=[ones, -ones], axis=0) - np.log(2) #If you encounter an overflow error here, you messed up big time. Please try another implementation or write your own.

def logcosh(x):
    """
    Auxiliary function
    Computes the natural logarithm of the cosh function, bypassing overflows that would happen if the sinh would be directly computed.
    """
    ones=np.ones_like(x)
    return logsumexp([x, -x], b=[ones, ones], axis=0) - np.log(2) #If you encounter an overflow error here, you messed up big time. Please try another implementation or write your own.

def calculateSmallV(a,b,c,d,N,V0,x_i,g):
    """
    a = Half the width of substrate
    d = Half the width of microstrip
    b = Height of entire geometry
    c = Thickness of substrate
    N = Number of fourier coefficents
    V0 = Potential of microstirp
    x_i = Location of potentials g (excluding x=d and x=a)
    g = Potentials g (exluding g(d)=V0 and g(a)=0)
    
    Calculates the fourier coefficients v_n.
    Consult WJ for more details on the problem.
    """
    if np.size(x_i)!=np.size(g):
        raise Exception('Dimensions of x_i and g have to agree!')
    if x_i[0]==0:
        x_i=np.delete(x_i,0)
        g=np.delete(g,0)
        #print('Deleting the first element because it contains 0 as a spline knot.')
    if x_i[0]==d:
        x_i=np.delete(x_i,0)
        g=np.delete(g,0)
        #print('Deleting the first element because it contains d as a spline knot.')
    if x_i[-1]==a:
        x_i=np.delete(x_i,-1)
        g=np.delete(g,-1)
        #print('Deleting the last element because it contains a as a spline knot.')
    n=np.linspace(0,N,N+1)
    m=np.size(x_i)

    # Making sure x_i and g are sorted vectors
    x_i=np.sort(x_i)
    if m==1: #Only one spline knot
        raise Exception('Not enough spline knots. Make sure there are at least 2 between d and a!')
    if m==2: #Python's slicing makes this necessary
        v_n0=2/a*V0*(1/((2*n+1)*np.pi/(2*a))**2)

        v_n1=(g[0]-V0)/(x_i[0]-d)*(np.cos((2*n+1)*x_i[0]*np.pi/(2*a))-np.cos((2*n+1)*d*np.pi/(2*a)))
        v_n3=g[1]/(a-x_i[1])*np.cos((2*n+1)*x_i[1]*np.pi/(2*a))

        x_i_t=np.reshape(x_i,(-1,1)) # Turning the 1 x m vector into a m x 1 vector
        cos1=np.cos(x_i_t[1]*np.pi/(2*a)*(2*n+1))
        cos2=np.cos(x_i_t[0]*np.pi/(2*a)*(2*n+1))
        v_n2=(g[1]-g[0])/(x_i[1]-x_i[0])*(cos1-cos2)

        v_n=v_n0*(v_n1+v_n2+v_n3)

        return v_n

    v_n0=2/a*V0*(1/((2*n+1)*np.pi/(2*a))**2)
    v_n1=(g[0]-V0)/(x_i[0]-d)*(np.cos((2*n+1)*x_i[0]*np.pi/(2*a))-np.cos((2*n+1)*d*np.pi/(2*a)))
    v_n3=g[m-1]/(a-x_i[m-1])*np.cos((2*n+1)*x_i[m-1]*np.pi/(2*a))

    x_i_t=np.reshape(x_i,(-1,1)) # Turning the 1 x m vector into a m x 1 vector

    cos1=np.cos(x_i_t[1:m]*np.pi/(2*a)*(2*n+1)) # m-1 x 1 * 1 x N = m-1 x N
    cos2=np.cos(x_i_t[0:m-1]*np.pi/(2*a)*(2*n+1))
    fac1=(g[1:m]-g[0:m-1])/(x_i[1:m]-x_i[0:m-1])
    v_n2=np.matmul(fac1,(cos1-cos2)) # 1 x m-1 * m-1 x N = 1 x N

    v_n=v_n0*(v_n1+v_n2+v_n3)

    return v_n

def calculateW(v_n,a,b,c,N,eps_r1,eps_r2):
    """
    ALL VARIABLES NEED TO BE IN SI UNITS
    v_n = Fourier coefficients computed by calculateSmallV
    a = Half the width of substrate
    b = Height of the entire geometry
    c = Thickness of substrate
    N = Number of fourier coefficients
    Consult WJ for more details on the problem

    Returns the energy of the field in one half of the geometry times two.
    """
    eps_1=eps_r1*8.854E-12
    eps_2=eps_r2*8.854E-12
    n=np.linspace(0,N,N+1)
    # Calculate first sum

    cosh1=logcosh((2*n+1)*np.pi*(b-c)/(2*a)) # 1 x N
    sinh1=logsinh((2*n+1)*np.pi*(b-c)/(2*a)) # 1 x N
    fac1 =(2*n+1)*np.pi*v_n**2 # 1 x N

    A=np.exp(cosh1-sinh1)
    sum1=eps_1/4*np.sum(fac1*A)

    # Calculate second sum

    cosh2=logcosh((2*n+1)*np.pi*c/(2*a)) # 1 x N
    sinh2=logsinh((2*n+1)*np.pi*c/(2*a)) # 1 x N
    fac2 =(2*n+1)*np.pi*v_n**2 # 1 x N

    B=np.exp(cosh2-sinh2)
    sum2=eps_2/4*np.sum(fac2*B)

    W=sum1+sum2
    #print('Energy of the half geometry in J for '+str(eps_r2) +': '+str(W))
    return W

def line_equ(x_start,x_intercept,x):
    m = -1/(x_intercept-x_start)
    b = 0 - m*x_intercept
    y = m*x + b
    return y

# check for necessary conditions (monotone decreasing convex)
def monotonous_convex_constraint(x_values, y_values):
    slopes = np.diff(y_values) / np.diff(x_values)
    return slopes


#------------------------------------------------------------------------------------------------------------------------
# Define single-objective Optimization Problem
#------------------------------------------------------------------------------------------------------------------------
class GA_Optimizer(Problem):

    def __init__(self,a,b,c,d,N,V0,x_i,eps_r1,eps_r2,lower,upper):
        super().__init__(n_var=len(x_i)-3,
                         n_obj=1,
                         n_ieq_constr=1,
                         xl = lower,
                         xu = upper)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.N = N
        self.V0 = V0
        self.x_i = x_i
        self.eps_r1 = eps_r1
        self.eps_r2 = eps_r2

    def _evaluate(self, x, out, *args, **kwargs):
        g_points = np.concatenate((np.ones((len(x),2)),x,np.zeros((len(x),1))),axis=1)
        
        W_array = np.zeros(len(x))
        for i in range(len(x)):
            v_n = calculateSmallV(self.a,self.b,self.c,self.d,self.N,self.V0,self.x_i,g_points[i,:])
            W = calculateW(v_n,self.a,self.b,self.c,self.N,self.eps_r1,self.eps_r2)
            W_array[i] = W
        
        violation = np.zeros(len(x))
        for i in range(len(x)):
            violation[i] = np.max(monotonous_convex_constraint(self.x_i[1:-1],g_points[i,1:-1]))-1

        out["F"] = W_array

        out["G"] = [violation]