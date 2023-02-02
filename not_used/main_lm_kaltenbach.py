import numpy as np
from numpy import Inf
import scipy as sp
from scipy import linalg

def givens_rotation(A, i, j):
    n, m = A.shape

    r = (A[j,j]**2 + A[i,j]**2) ** 0.5
    c = A[j,j]/r
    s = A[i,j]/r

    # Q: Givens rotation matrix
    Q = np.eye(n)
    Q[j,j] = c
    Q[i,i] = c
    Q[i,j] = -s
    Q[j,i] = s

    # B = QA, B[i,j] = 0
    B = A.astype(float)
    S = np.array([[c,s], [-s,c]])
    C = A[(j,i),:]
    D = np.dot(S, C)
    B[j] = D[0]
    B[i] = D[1]

    return B, Q

def givens_qr(R_I, n, m):

    # list for the following loop
    rows = list(range(m+n, m, -1))

    # l: num of rotations required in each row
    l = 1

    # Q stores all the rotations
    Q = np.eye(n+m)

    # proceed along pp. 14-15
    for k in rows:
        for i in list(range(min(l,n), 0, -1)):
            R_I, Q_1 = givens_rotation(R_I, k-1, n-i)
            Q = np.dot(Q_1, Q)

        l += 1
    
    return R_I, Q.T

def LM(
    r,              # residual vector r(x), dim: m
    J,              # Jacobian matrix, dim: mxn
    x,              # init. point, dim: n
    Delta=1e2,
    Delta_max=1e4,
    eta=1e-4,
    sigma=1e-1,
    nmax=500,
    tol_abs=1e-7,
    tol_rel=1e-7,
    eps=1e-3,
    Scaling=False
):
    def f(x):
        return 0.5*np.linalg.norm(r(x))**2
    def gradf(x):
        return J(x).T @ r(x)
    
    counter = 0
    fx = f(x)
    func_eval = 1
    m, n = J(x).shape

    Information = [['counter', 'norm of step p', 'x', 'norm of the gradient at x']]
    abs_grad_f = np.linalg.norm(gradf(x))
    Information.append([counter, 'not available', x, abs_grad_f])

    tolerance = min((tol_rel * Information[-1][-1] + tol_abs), eps)

    D = np.eye(n)
    D_inv = np.eye(n)

    while abs_grad_f > tolerance and counter < nmax:
        Jx = J(x)

        if Scaling:
            for i in range(n):
                D[i,i] = max(D[i,i], np.linalg.norm(Jx[:,i]))
                D_inv[i,i] = 1/D[i,i]
        D_2 = D @ D

        Q, R, Pi = sp.linalg.qr(Jx, pivoting=True)
        P = np.eye(n)[:,Pi]

        rank = np.linalg.matrix_rank(Jx)
        if rank == n:
            p = P @ sp.linalg.solve_triangular(R[:n,:], np.dot(Q[:,:n].T, -r(x)))
        else:
            y = np.zeros(n)
            y[:rank] = sp.linalg.solve_triangular(R[:rank,:rank], Q[:,:rank].T @ (-r(x)))
            p = P @ y
        
        Dp = np.linalg.norm(D @ p)

        if Dp <= ((1+sigma)*Delta):
            alpha = 0
        else:
            J_scaled = Jx @ D_inv
            u = np.linalg.norm(J_scaled.T @ r(x)) / Delta
            if rank == n:
                q = sp.linalg.solve_triangular(R[:n,:].T, P.T @ D_2 @ p, lower=True)
                l = (Dp - Delta) / (np.linalg.norm(q)**2/Dp)
            else:
                l = 0
            
            if u == Inf:
                alpha = 1
            else:
                alpha = max(1e-3*u, (l*u)**0.5)
            
            while Dp > (1+sigma)*Delta or Dp < (1-sigma)*Delta:
                if alpha == Inf:
                    print('Error: LM fails (lambda too large), change x0')
                    return x, Information
                
                if alpha <= 1 or alpha > u:
                    alpha = max(1e-3*u, (l*u)**0.5)

                D_lambda = P.T @ D @ P
                R_I = np.concatenate((R, alpha**0.5 * D_lambda), axis=0)

                R_lambda, Q_lambda2 = givens_qr(R_I, n, m)

                Q_lambda = np.concatenate((np.concatenate((Q, np.zeros((m,n))),axis=1),
                                          np.concatenate((np.zeros((n,m)), P), axis=1)),
                                          axis=0) @ Q_lambda2
                
                r_0 = np.append(r(x), np.zeros(n))

                p = P @ sp.linalg.solve_triangular(R_lambda[:n,:], Q_lambda[:,:n].T @ (-r_0))
                Dp = np.linalg.norm(D@p)

                q = sp.linalg.solve_triangular(R_lambda[:n,:].T, P.T @ D_2 @ p, lower=True)
                phi = Dp - Delta
                phi_derivative = -np.linalg.norm(q)**2 / Dp

                if phi < 0:
                    u = alpha
                l = max(l, alpha - phi/phi_derivative)

                alpha = alpha - ((phi+Delta)/Delta) * (phi/phi_derivative)

        fxp = f(x+p)
        func_eval += 1

        if fxp > fx or fxp == Inf or np.isnan(fxp):
            rho = 0
        else:
            ared = 1 - fxp/fx
            pred = (0.5*np.linalg.norm(Jx@p)**2)/fx + (alpha*Dp**2)/fx
            rho = ared/pred

        if rho < 0.25:
            Delta = 0.25 * Delta
        else:
            if rho > 0.75 and (Dp >= (1-sigma)*Delta):
                Delta = min(2*Delta, Delta_max)
        
        if rho > eta:
            x += p
            fx = fxp
            counter += 1
            Information.append([counter, np.linalg.norm(p), x, np.linalg.norm(gradf(x))])
    
    if Information[-1][-1] <= tolerance:
        print(f'LM terminated successfully, f(x): {fx:.4f},\titer: {counter},\tfunc eval: {func_eval}')
    else:
        print(f'LM failed to converge within {nmax} steps...')
    
    return x, Information

def main():

    def r(x):
        return np.r_[2**0.5 * (1-x[0]), 10 * 2**0.5 * (x[1] - x[0]**0.5)]

    def J(x):
        return np.r_[[[-2**0.5, 0],
                      [-20*2**0.5*x[0], 10*2**0.5]]]

    x0 = np.r_[0.1,-0.1]
    x, Info = LM(r, J, x0)

    print('hi')


if __name__ == "__main__":
	main()