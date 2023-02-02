import numpy as np
from numpy.linalg import inv
from numpy import Inf
import scipy as sp
from scipy import linalg
# from params import *

iteration = 0
func_calls = 0

def lm_func(t, p):
    global func_calls
    func_calls += 1
    return p[0]*np.exp(-t/p[1]) + p[2]*t * np.exp(-t/p[3])

# J = lm_FD_J(func,t,p,y,{dp},{c})
#
# partial derivatives (Jacobian) dy/dp for use with lm.m
# computed via Finite Differences
# Requires n or 2n function evaluations, n = number of nonzero values of dp
# -------- INPUT VARIABLES ---------
# func = function of independent variables, 't', and parameters, 'p',
#        returning the simulated model: y_hat = func(t,p,c)
# t  = independent variables (used as arg to func)                       (m x 1)
# p  = current parameter values                                          (n x 1)
# y  = func(t,p,c) initialised by user before each call to lm_FD_J       (m x 1)
# dp = fractional increment of p for numerical derivatives
#      dp(j)>0 central differences calculated
#      dp(j)<0 one sided differences calculated
#      dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
#      Default:  0.001;
# c  = optional vector of constants passed to y_hat = func(t,p,c)
#---------- OUTPUT VARIABLES -------
# J  = Jacobian Matrix J(i,j)=dy(i)/dp(j)         i=1:n; j=1:m
def J_FD(func, t, p, y, dp):
    m = len(y)
    n = len(p)

    ps = np.copy(p)
    J = np.zeros((m,n))
    delta = np.zeros((n,1))

    for j in range(n):
        delta[j] = dp[j] * (1 + np.abs(p[j]))
        p[j] = ps[j] + delta[j]
        if delta[j] != 0:
            y1 = func(t, p)

            if dp[j] < 0:
                J[:,j] = (y1 - y)/delta[j]
            else:
                p[j] = ps[j] - delta[j]
                J[:,j] = (y1 - func(t, p)) / (2 * delta[j])
        
        p[j] = ps[j]    # don't know the role of this line
    
    return J

# J = lm_Broyden_J(p_old,y_old,J,p,y)
# carry out a rank-1 update to the Jacobian matrix using Broyden's equation
#---------- INPUT VARIABLES -------
# p_old = previous set of parameters                                     (n x 1)
# y_old = model evaluation at previous set of parameters, y_hat(t;p_old) (m x 1)
# J  = current version of the Jacobian matrix                            (m x n)
# p     = current  set of parameters                                     (n x 1)
# y     = model evaluation at current  set of parameters, y_hat(t;p)     (m x 1)
#---------- OUTPUT VARIABLES -------
# J = rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j)  i=1:n; j=1:m  (m x n)
def J_Broyden(p_old, y_old, J, p, y):
    h = p - p_old
    return J + ((y- y_old - J@h).reshape(-1,1) @ h.reshape(1,-1)) / (h.T @ h)


# Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, 
# and calculate the Chi-squared error function, Chi_sq 
# Used by Levenberg-Marquard algorithm, lm.m   
# -------- INPUT VARIABLES ---------
# func   = function ofpn independent variables, p, and m parameters, p,
#         returning the simulated model: y_hat = func(t,p,c)
# t      = independent variables (used as arg to func)                   (m x 1)
# p_old  = previous parameter values                                     (n x 1)
# y_old  = previous model ... y_old = y_hat(t;p_old);                    (m x 1)
# dX2    = previous change in Chi-squared criteria                       (1 x 1)
# J      = Jacobian of model, y_hat, with respect to parameters, p       (m x n)
# p      = current  parameter values                                     (n x 1)
# y_dat  = data to be fit by func(t,p,c)                                 (m x 1)
# weight = the weighting vector for least squares fit ...
#          inverse of the squared standard measurement errors
# dp     = fractional increment of 'p' for numerical derivatives
#          dp(j)>0 central differences calculated
#          dp(j)<0 one sided differences calculated
#          dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
#          Default:  0.001;
# c      = optional vector of constants passed to y_hat = func(t,p,c)
#---------- OUTPUT VARIABLES -------
# JtWJ    = linearized Hessian matrix (inverse of covariance matrix)     (n x n)
# JtWdy   = linearized fitting vector                                    (n x m)
# Chi_sq = Chi-squared criteria: weighted sum of the squared residuals WSSR
# y_hat  = model evaluated with parameters 'p'                           (m x 1)
# J      = Jacobian of model, y_hat, with respect to parameters, p       (m x n)
def core_operation(func, t, p_old, y_old, dX2, J, p, y_dat, weight, dp):

    Npar = len(p)

    y_hat = func(t, p)
    global iteration
    if iteration%(2*Npar) == 0 or dX2 > 0:
        J = J_FD(func, t, p, y_hat, dp)
    else:
        J = J_Broyden(p_old, y_old, J, p, y_hat)
    
    delta_y = y_dat - y_hat
    Chi_sq = delta_y.T @ np.multiply(delta_y, weight)
    JtWJ = J.T @ (np.multiply(J, weight.reshape(-1,1) @ np.ones((1, Npar))))
    JtWdy = J.T @ (np.multiply(weight, delta_y))

    return JtWJ, JtWdy, Chi_sq, y_hat, J



# Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals
# ----------  INPUT  VARIABLES  -----------
# func   = function of n independent variables, 't', and m parameters, 'p', 
#          returning the simulated model: y_hat = func(t,p,c)
# p      = initial guess of parameter values                             (n x 1)
# t      = independent variables (used as arg to func)                   (m x 1)
# y_dat  = data to be fit by func(t,p)                                   (m x 1)
# weight = weights or a scalar weight value ( weight >= 0 ) ...          (m x 1)
#          inverse of the standard measurement errors
#          Default:  ( 1 / ( y_dat' * y_dat ))
# dp     = fractional increment of 'p' for numerical derivatives
#          dp(j)>0 central differences calculated
#          dp(j)<0 one sided 'backwards' differences calculated
#          dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
#          Default:  0.001;
# p_min  = lower bounds for parameter values                             (n x 1)
# p_max  = upper bounds for parameter values                             (n x 1)
# opts   = vector of algorithmic parameters
#             parameter    defaults    meaning
# opts(1)  =  prnt            3        >1 intermediate results; >2 plots
# opts(2)  =  MaxIter      10*Npar     maximum number of iterations
# opts(3)  =  epsilon_1       1e-3     convergence tolerance for gradient
# opts(4)  =  epsilon_2       1e-3     convergence tolerance for parameters
# opts(5)  =  epsilon_3       1e-1     convergence tolerance for red. Chi-square
# opts(6)  =  epsilon_4       1e-1     determines acceptance of a L-M step
# opts(7)  =  lambda_0        1e-2     initial value of L-M paramter
# opts(8)  =  lambda_UP_fac   11       factor for increasing lambda
# opts(9)  =  lambda_DN_fac    9       factor for decreasing lambda
# opts(10) =  Update_Type      1       1: Levenberg-Marquardt lambda update
#                                      2: Quadratic update 
#                                      3: Nielsen's lambda update equations
#
# ----------  OUTPUT  VARIABLES  -----------
# p       = least-squares optimal estimate of the parameter values
# redX2   = reduced Chi squared error criteria - should be close to 1
# sigma_p = asymptotic standard error of the parameters
# sigma_y = asymptotic standard error of the curve-fit
# corr_p  = correlation matrix of the parameters
# R_sq    = R-squared cofficient of multiple determination  
# cvg_hst = convergence history ... see lm_plots.m
def lm(func, p, t, y_dat, weight=4, dp=1e-3, p_min=-50*np.ones(4), p_max=50*np.ones(4), \
    opts=None):
    
    global iteration
    iteration = 0
    Npar = len(p)
    Npnt = len(y_dat)
    p_old = np.zeros(Npar)
    y_old = np.zeros(Npnt)
    X2 = np.inf         # big initial X2 value
    X2_old = np.inf     # big initial X2 value
    J = np.zeros((Npnt, Npar))
    DoF = Npnt - Npar
    stop = False

    if len(t) != Npnt:
        print('Error: num of t must equal to the len of y_dat...')
        return
    
    if opts is None:
                #   prnt MaxIter  eps1  eps2  eps3  eps4  lam0  lamUP lamDN UpdateType
        opts = np.r_[  1,10*Npar, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2,    11,    9,        1]
    prnt, MaxIter, \
        epsilon_1, epsilon_2, epsilon_3, epsilon_4, \
            lambda_0, lambda_UP_fac, lambda_DN_fac, Update_Type = opts
    MaxIter = np.int64(MaxIter)
    
    if np.isscalar(dp):
        dp = dp * np.ones(Npar)
    
    if np.isscalar(weight):     # using uniform weights for error analysis
        weight = np.abs(weight) * np.ones(Npnt)
    
    JtWJ, JtWdy, X2, y_hat, J = core_operation(func, t, p_old, y_old, 1, J, p, y_dat, weight, dp)

    if np.max(np.abs(JtWdy)) < epsilon_1:
        print('init. guess is extremely close to the optimal')
        print(f'epsilon_1 = {epsilon_1:f}')
        stop = True
    
    ll = lambda_0   # init lambda

    X2_old = X2
    cvg_hst = np.ones((MaxIter, Npar+3))    #  convergence history
    while not stop and iteration <= MaxIter:
        iteration += 1
        h = inv(JtWJ + ll * np.diag(np.diag(JtWJ))) @ JtWdy
        p_try = p + h
        p_try = np.min((np.max((p_min, p_try), axis=0), p_max), axis=0)

        delta_y = y_dat - func(t, p_try)
        X2_try = delta_y.T @ np.multiply(weight, delta_y)
        rho = (X2 - X2_try) / (np.abs(h.T @ (ll*np.diag(np.diag(JtWJ)) @ h + JtWdy)))
        
        # update if is significantly better
        if (rho > epsilon_4):
            dX2 = X2 - X2_old
            X2_old = X2
            p_old = p
            y_old = y_hat
            p = p_try

            JtWJ, JtWdy, X2, y_hat, J = \
                core_operation(func, t, p_old, y_old, dX2, J, p, y_dat, weight, dp)

            ll = np.max((ll / lambda_DN_fac, 1e-7))
        else:
            X2 = X2_old
            if (iteration % 2*Npar) == 0:
                JtWJ, JtWdy, dX2, y_hat, J = \
                    core_operation(func, t, p_old, y_old, -1, J, p, y_dat, weight, dp)
            ll = np.min((ll * lambda_UP_fac, 1e7))
        
        if prnt > 1:
            print(f'iter {iteration}, func_call {func_calls}, chi_sq {X2/DoF:.3f}, lambda {ll:.3f}')
            print(f'param {p}')
            print(f'dp/p {h/p}')            
        
        cvg_hst[iteration, :] = np.r_[func_calls, p, X2/DoF, ll]

        if iteration > 2:
            if np.max(np.abs(JtWdy)) < epsilon_1:
                print(f'convergence in r.h.s. (JtWdy), epsilon_1 = {epsilon_1}')
                stop = True
            if np.max(np.abs(h) / np.abs(p) + 1e-12) < epsilon_2:
                print(f'convergence in params, epsilon_2 = {epsilon_2}')
                stop = True
            if X2/DoF < epsilon_3:
                print(f'converged, epsilon_1 = {epsilon_1}')
                stop = True
        if iteration == MaxIter:
            print('Max iteration without convergence...')
            stop=True
    
    if np.min(weight) == np.max(weight):
        weight = DoF / (delta_y @ delta_y) * np.ones(Npnt)
    
    redX2 = X2 / DoF

    JtWJ, JtWdy, X2, y_hat, J = \
        core_operation(func, t, p_old, y_old, -1, J, p, y_dat, weight, dp)
    
    covar_p = inv(JtWJ)
    sigma_p = np.sqrt(np.diag(covar_p))
    
    sigma_y = np.zeros(Npnt)
    for i in range(Npnt):
        sigma_y[i] = J[i,:] @ covar_p @ J[i,:].T
    sigma_y = np.sqrt(sigma_y)

    corr_p = covar_p / (sigma_p @ sigma_p.T)

    R_sq = np.corrcoef(y_dat, y_hat)
    R_sq = R_sq[0,1] ** 2

    cvg_hst = cvg_hst[:iteration, :]

    return p, redX2, sigma_p, sigma_y, corr_p, R_sq, cvg_hst

def main():

    np.random.seed(0)
    f = lm_func
    p_true = np.r_[20, 10, 1, 50]
    
    Npnt = 100

    sigma_n = 0.5   # measurement noise

    t = np.arange(Npnt) + 1
    y_dat = lm_func(t, p_true)
    y_dat += sigma_n*np.random.random(y_dat.shape)    

    p_init = np.r_[5, 2, 0.2, 10]
    weight = 1/sigma_n**2

    # p_min = np.r_[-50, -50, -50, -50]
    # p_max = np.r_[50, 50, 50, 50]
    p_min = -10*np.abs(p_init)
    p_max = 10*np.abs(p_init)
            #  prnt MaxIter  eps1  eps2  eps3  eps4  lam0  lamUP lamDN UpdateType
    opts = np.r_[  1,    100, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2,    11,    9,        1]

    p_fit, Chi_sq, sigma_p, sigma_y, corr, R_sq, cvg_hst = \
        lm(f, p_init, t, y_dat, weight, -0.01, p_min, p_max, opts)

    y_fit = lm_func(t, p_fit)




    print('hi')


if __name__ == "__main__":
	main()