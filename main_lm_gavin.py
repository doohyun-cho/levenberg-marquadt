import numpy as np
from lm_gavin_util import LM
from matplotlib import pyplot as plt

np.random.seed(0)

def lm_func(t, p):
    return p[0]*np.exp(-t/p[1]) + p[2]*t * np.exp(-t/p[3])

def lm_func1(t, p):
    return np.r_[(1-p[0])**2 + 100*(p[1]-p[0]**2)**2]
    
def lm_func2(t, p): 
    return np.r_[np.sqrt(2)*(1-p[0]), 10*np.sqrt(2) * (p[1] - p[0]**2)]
    
def lm_func3(t, x): 
    return np.r_[np.sqrt(2)*(x[0]**2 + x[1] - 11), np.sqrt(2) * (x[0] + x[1]**2 - 7)]

def lm_func4(t, x):
    return np.r_[x[0]**3+5, x[1]**3 - 4]

def main():
    lm = LM()

    if False:
        f = lm_func
        x_init = np.r_[5, 2, 0.2, 10]
        
        Npnt = 100
        t = np.arange(Npnt) + 1
        x_true = np.r_[20, 10, 1, 50]
        f_dat = f(t, x_true)
        sigma_n = 0.5   # measurement noise
        f_dat += sigma_n*np.random.random(f_dat.shape)    
        weight = 1/sigma_n**2

        x_min = np.r_[-50, -50, -50, -50]
        x_max = np.r_[50, 50, 50, 50]

    if True:
        f = lm_func4
        # p_init = np.r_[0, 0]
        # p_init = np.r_[1.1,0.9]
        x_init = np.r_[-0.1, 0.1]
        
        Npnt = len(x_init)
        t = np.arange(Npnt)
        f_dat = 0*np.ones(x_init.shape)    # wanna set residals to be zero
        sigma_n = 0.5   # measurement noise
        # y_dat += sigma_n*np.random.random(y_dat.shape)    
        weight = 1/sigma_n**2

        x_min = -10*np.ones(x_init.shape)
        x_max = 10*np.ones(x_init.shape)

    # y_dat = np.zeros(t.shape)    # to minimize
            #  prnt MaxIter  eps1  eps2  eps3  eps4  lam0  lamUP lamDN UpdateType
    opts = np.r_[  2,    1000, 1e-4, 1e-3, 1e-3, 1e-1, 1e-2,    11,    9,        1]

    p_fit, Chi_sq, sigma_p, sigma_y, corr, R_sq, cvg_hst = \
        lm.lm(f, x_init, t, f_dat, weight, -0.01, x_min, x_max, opts)

    f_fit = f(t, p_fit)

    # plt.figure()
    # plt.scatter(t, y_dat)
    # plt.plot(t, y_fit)

    print('hi')

if __name__ == "__main__":
	main()