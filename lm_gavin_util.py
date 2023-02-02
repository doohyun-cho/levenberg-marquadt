import numpy as np
from numpy.linalg import inv
from numpy import Inf
import scipy as sp
from scipy import linalg

class LM():

    def __init__(self) -> None:
        self.iteration = 0
        self.func_calls = 0

    # J = J_FD(func,t,x,f,{dx})
    #
    # partial derivatives (Jacobian) dy/dx for use with lm.m
    # computed via Finite Differences
    # Requires n or 2n function evaluations, n = number of nonzero values of dx
    # -------- INPUT VARIABLES ---------
    # func = function of independent variables, 't', and parameters, 'x',
    #        returning the simulated model: y_hat = func(t,x)
    # t  = independent variables (used as arg to func)                       (m x 1)
    # x  = current parameter values                                          (n x 1)
    # f  = func(t,x) initialised by user before each call to lm_FD_J       (m x 1)
    # dx = fractional increment of x for numerical derivatives
    #      dx(j)>0 central differences calculated
    #      dx(j)<0 one sided differences calculated
    #      dx(j)=0 sets corresponding partials to zero; i.e. holds x(j) fixed
    #      Default:  0.001;
    #---------- OUTPUT VARIABLES -------
    # J  = Jacobian Matrix J(i,j)=dy(i)/dx(j)         i=1:n; j=1:m
    def J_FD(self, func, t, x, f, dx):
        m = len(f)
        n = len(x)

        ps = np.copy(x)
        J = np.zeros((m,n))
        delta = np.zeros((n,1))

        for j in range(n):
            delta[j] = dx[j] * (1 + np.abs(x[j]))
            x[j] = ps[j] + delta[j]
            if delta[j] != 0:
                y1 = func(t, x)

                if dx[j] < 0:
                    J[:,j] = (y1 - f)/delta[j]
                else:
                    x[j] = ps[j] - delta[j]
                    J[:,j] = (y1 - func(t, x)) / (2 * delta[j])
            
            x[j] = ps[j]    # don't know the role of this line
        
        return J

    # J = lm_Broyden_J(x_old,f_old,J,x,f)
    # carry out a rank-1 update to the Jacobian matrix using Broyden's equation
    #---------- INPUT VARIABLES -------
    # x_old = previous set of parameters                                     (n x 1)
    # f_old = model evaluation at previous set of parameters, y_hat(t;x_old) (m x 1)
    # J  = current version of the Jacobian matrix                            (m x n)
    # x     = current  set of parameters                                     (n x 1)
    # f     = model evaluation at current  set of parameters, y_hat(t;x)     (m x 1)
    #---------- OUTPUT VARIABLES -------
    # J = rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dx(j)  i=1:n; j=1:m  (m x n)
    def J_Broyden(self, x_old, f_old, J, x, f):
        h = x - x_old
        return J + ((f - f_old - J@h).reshape(-1,1) @ h.reshape(1,-1)) / (h.T @ h)


    # Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, 
    # and calculate the Chi-squared error function, Chi_sq 
    # Used by Levenberg-Marquard algorithm, lm.m   
    # -------- INPUT VARIABLES ---------
    # func   = function ofpn independent variables, x, and m parameters, x,
    #         returning the simulated model: y_hat = func(t,x)
    # t      = independent variables (used as arg to func)                   (m x 1)
    # x_old  = previous parameter values                                     (n x 1)
    # f_old  = previous model ... f_old = y_hat(t;x_old);                    (m x 1)
    # dX2    = previous change in Chi-squared criteria                       (1 x 1)
    # J      = Jacobian of model, y_hat, with respect to parameters, x       (m x n)
    # x      = current  parameter values                                     (n x 1)
    # y_dat  = data to be fit by func(t,x)                                   (m x 1)
    # weight = the weighting vector for least squares fit ...
    #          inverse of the squared standard measurement errors
    # dx     = fractional increment of 'x' for numerical derivatives
    #          dx(j)>0 central differences calculated
    #          dx(j)<0 one sided differences calculated
    #          dx(j)=0 sets corresponding partials to zero; i.e. holds x(j) fixed
    #          Default:  0.001;
    #---------- OUTPUT VARIABLES -------
    # JtWJ    = linearized Hessian matrix (inverse of covariance matrix)     (n x n)
    # JtWdy   = linearized fitting vector                                    (n x m)
    # Chi_sq = Chi-squared criteria: weighted sum of the squared residuals WSSR
    # y_hat  = model evaluated with parameters 'x'                           (m x 1)
    # J      = Jacobian of model, y_hat, with respect to parameters, x       (m x n)
    def core_operation(self, func, t, x_old, f_old, dX2, J, x, f_dat, weight, dx):

        Npar = len(x)

        f_hat = func(t, x)
        if self.iteration%(2*Npar) == 0 or dX2 > 0:
            J = self.J_FD(func, t, x, f_hat, dx)
        else:
            J = self.J_Broyden(x_old, f_old, J, x, f_hat)
        
        df = f_dat - f_hat
        X2 = df.T @ np.multiply(df, weight)
        JtWJ = J.T @ (np.multiply(J, weight.reshape(-1,1) @ np.ones((1, Npar))))
        JtWdf = J.T @ (np.multiply(weight, df))

        return JtWJ, JtWdf, X2, f_hat, J

    # Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals
    # ----------  INPUT  VARIABLES  -----------
    # func   = function of n independent variables, 't', and m parameters, 'x', 
    #          returning the simulated model: y_hat = func(t,x)
    # x      = initial guess of parameter values                             (n x 1)
    # t      = independent variables (used as arg to func)                   (m x 1)
    # y_dat  = data to be fit by func(t,x)                                   (m x 1)
    # weight = weights or a scalar weight value ( weight >= 0 ) ...          (m x 1)
    #          inverse of the standard measurement errors
    #          Default:  ( 1 / ( y_dat' * y_dat ))
    # dx     = fractional increment of 'x' for numerical derivatives
    #          dx(j)>0 central differences calculated
    #          dx(j)<0 one sided 'backwards' differences calculated
    #          dx(j)=0 sets corresponding partials to zero; i.e. holds x(j) fixed
    #          Default:  0.001;
    # x_min  = lower bounds for parameter values                             (n x 1)
    # x_max  = upper bounds for parameter values                             (n x 1)
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
    # x       = least-squares optimal estimate of the parameter values
    # redX2   = reduced Chi squared error criteria - should be close to 1
    # sigma_x = asymptotic standard error of the parameters
    # sigma_y = asymptotic standard error of the curve-fit
    # corr_x  = correlation matrix of the parameters
    # R_sq    = R-squared cofficient of multiple determination  
    # cvg_hst = convergence history ... see lm_plots.m
    def lm(self, f_, x, t, f_dat, weight=4, dx=1e-3, x_min=-50*np.ones(4), x_max=50*np.ones(4), opts=None):

        def f(*args):
            self.func_calls += 1
            return f_(*args)

        self.iteration = 0
        self.func_calls = 0
        len_x = len(x)
        len_f = len(f_dat)
        x_old = np.zeros(len_x)
        f_old = np.zeros(len_f)
        X2 = np.inf         # big initial X2 value
        X2_old = np.inf     # big initial X2 value
        J = np.zeros((len_f, len_x))
        DoF = np.max((len_f - len_x, 1))
        stop = False

        if len(t) != len_f:
            print('Error: num of t must equal to the len of y_dat...')
            return
        
        if opts is None:
                    #   prnt MaxIter  eps1  eps2  eps3  eps4  lam0  lamUP lamDN UpdateType
            opts = np.r_[  1,10*len_x, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2,    11,    9,        1]
        prnt, MaxIter, eps1, eps2, eps3, eps4, ll0, ll_up, ll_down, Update_Type = opts
        MaxIter = np.int64(MaxIter)
        
        if np.isscalar(dx):
            dx = dx * np.ones(len_x)
        
        if np.isscalar(weight):     # using uniform weights for error analysis
            weight = np.abs(weight) * np.ones(len_f)
        
        JtWJ, JtWdy, X2, f_hat, J = self.core_operation(f, t, x_old, f_old, 1, J, x, f_dat, weight, dx)

        if np.max(np.abs(JtWdy)) < eps1:
            print('init. guess is extremely close to the optimal')
            print(f'epsilon_1 = {eps1:f}')
            stop = True
        
        ll = ll0   # init lambda

        X2_old = X2
        cvg_hst = np.ones((MaxIter, len_x+3))    #  convergence history
        df = f(t, x) - f_dat
        print(f'iter  f_call    chi_sq    lambda      dx/x           x')
        while not stop and self.iteration < MaxIter:
            self.iteration += 1
            h = inv(JtWJ + ll * np.diag(np.diag(JtWJ))) @ JtWdy
            x_try = x + h
            x_try = np.min((np.max((x_min, x_try), axis=0), x_max), axis=0)

            df = f(t, x_try) - f_dat
            X2_try = df.T @ np.multiply(weight, df)
            rho = (X2 - X2_try) / (np.abs(h.T @ (ll*np.diag(np.diag(JtWJ)) @ h + JtWdy)))
            
            # update if is significantly better
            if (rho > eps4):
                dX2 = X2 - X2_old
                X2_old = X2
                x_old = x
                f_old = f_hat
                x = x_try

                JtWJ, JtWdy, X2, f_hat, J = \
                    self.core_operation(f, t, x_old, f_old, dX2, J, x, f_dat, weight, dx)

                ll = np.max((ll / ll_down, 1e-7))
            else:
                X2 = X2_old
                if (self.iteration % 2*len_x) == 0:
                    JtWJ, JtWdy, dX2, f_hat, J = \
                        self.core_operation(f, t, x_old, f_old, -1, J, x, f_dat, weight, dx)
                ll = np.min((ll * ll_up, 1e7))
            
            if prnt > 1:
                print(f'{self.iteration:4d}  {self.func_calls:6d}  {X2/DoF:.2e}  {ll:.2e}  {np.linalg.norm(h/x):.2e}   {np.round(x, 3)}')
            
            cvg_hst[self.iteration, :] = np.r_[self.func_calls, x, X2/DoF, ll]

            if self.iteration > 2:
                if np.max(np.abs(JtWdy)) < eps1:
                    print(f'convergence in r.h.s. (JtWdy), epsilon_1 = {eps1}')
                    stop = True
                if np.max(np.abs(h) / np.abs(x) + 1e-12) < eps2:
                    print(f'convergence in params, epsilon_2 = {eps2}')
                    stop = True
                if X2/DoF < eps3:
                    print(f'convergence in X2, epsilon_3 = {eps3}')
                    stop = True
            if self.iteration == MaxIter:
                print('Max iteration without convergence...')
                stop=True
        
        if np.min(weight) == np.max(weight):
            weight = DoF / (df @ df) * np.ones(len_f)
        
        X2_normalized = X2 / DoF

        JtWJ, JtWdy, X2, f_hat, J = \
            self.core_operation(f, t, x_old, f_old, -1, J, x, f_dat, weight, dx)
        
        try:
            covar_p = inv(JtWJ)
            sigma_p = np.sqrt(np.diag(covar_p))
            
            sigma_y = np.zeros(len_f)
            for i in range(len_f):
                sigma_y[i] = J[i,:] @ covar_p @ J[i,:].T
            sigma_y = np.sqrt(sigma_y)

            corr_p = covar_p / (sigma_p @ sigma_p.T)

            R_sq = np.corrcoef(f_dat, f_hat)
            R_sq = R_sq[0,1] ** 2

            cvg_hst = cvg_hst[:self.iteration, :]

            return x, X2_normalized, sigma_p, sigma_y, corr_p, R_sq, cvg_hst
        except:
            return x
