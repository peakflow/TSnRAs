"""
Created on Mon May 06 14:46:09 2013, last update: 2013-05-06 (Mon)
@Filename: lq.py : doubleo.py, olrp.py, kfilter.py
@Author: Thomas Sargent, Doc-Jin Jang, Jeonghoon Choi, Younghwan Lee

"""

import numpy as np
from numpy import linalg as la
from numpy import dot, sqrt, abs, max, eye
from numpy.linalg import inv


def tri_dot(A,B,C): # A * B * C
    return dot(A,dot(B,C))

def l_div(A,B): # A^{-1} * B
    return dot(inv(A), B)

def r_div(A,B): # A * B^{-1}
    return dot(A, inv(B))



def doubleo(A, C, Q, R):
    """
    K, S = doubleo(A,C,Q,R)
    This program uses the "doubling algorithm" to solve the Riccati matrix 
    difference equations associated with the Kalman filter.  
    A is nxn, C is kxn, Q is nxn, R is kxk. 

    The program returns the gain K and the stationary covariance matrix 
    of the one-step ahead errors in forecasting the state.

    The program creates the Kalman filter for the following system:
           x(t+1) = A * x(t) + e(t+1)
             y(t) = C * x(t) + v(t)
    
    where E e(t+1)*e(t+1)' =  Q, and E v(t)*v(t)' = R, and v(s) is orthogonal
    to e(t) for all t and s. 
    
    The program creates the observer system
           xx(t+1) = A * xx(t) + K * a(t)
           y(t) = C * xx(t) + a(t),
    
    where K is the Kalman gain ,S = E (x(t) - xx(t))*(x(t) - xx(t))', and
    a(t) = y(t) - E[y(t)| y(t-1), y(t-2), ... ], and xx(t)=E[x(t)|y(t-1),...].
    
    NOTE:  By using DUALITY, control problems can also be solved.
    """
    a0 = A.T
    b0 = dot(C.T, l_div(R, C))
    g0 = Q
    
    tol, dd, ss = 1e-15, 1, max(A.shape)
    v = eye(ss)
    
    while dd > tol:
        a1 = dot(a0, l_div(v + dot(b0, g0), a0))
        b1 = b0 + dot(a0, l_div(v + dot(b0, g0), dot(b0, a0.T)))
        g1 = g0 + tri_dot(a0.T, g0, l_div(v + dot(b0, g0), a0))
        
        k1 = r_div(tri_dot(A, g1, C.T), (tri_dot(C, g1, C.T) + R))
        k0 = r_div(tri_dot(A, g0, C.T), (tri_dot(C, g0, C.T) + R))
        
        dd = max(abs(k1 - k0)) # dd = np.sort(abs(k1-k0))[0,-1]
        a0, b0, g0 = a1, b1, g1
    
    return k1, g1



def olrp(beta, A, B, Q, R, W=None):
    """
    [f, p] = olrp(beta, A, B Q, R, W)
    OLRP can have arguments: (beta,A,B,Q,R) if there are no cross products 
    (i.e. W=0).  Set beta=1, if there is no discounting.
    
    OLRP calculates f of the feedback law: 
        u = -fx
        
    that maximizes the function:
        sum {beta^t [x'Qx + u'Ru +2x'Wu] }
        
    subject to 
  	x[t+1] = Ax[t] + Bu[t] 
  
    where x is the nx1 vector of states, u is the kx1 vector of controls,
    A is nxn, B is nxk, Q is nxn, R is kxk, W is nxk.
                
    Also returned is p, the steady-state solution to the associated 
    discrete matrix Riccati equation.
    """
    m = max(A.shape)
    cb = B.shape[1]
    
    if W == None: W = np.zeros((m, cb))

    if np.min(abs(la.eig(R)[0])) > 1e-5: #if np.sort(abs(la.eig(R)[0]))[0] > 1e-5:
        A = sqrt(beta) * (A - tri_dot(B, inv(R), W.T))
        B = sqrt(beta) * B
        Q = Q - tri_dot(W, inv(R), W.T)
        
        k, s = doubleo(A.T, B.T, Q, R)
     
        return k.T + dot(inv(R), W.T), s
        
    else:
        p0 = -.01 * eye(m)
        dd, it, maxit = 1, 1, 1000

    # check tolerance; for greater accuracy set it to 1e-10
        while dd > 1e-6 and it <= maxit:
            f0 =  l_div((R + beta * tri_dot(B.T, p0, B)), (beta * dot(B.T, p0, A) + W.T))
            p1 = (beta * tri_dot(A.T, p0, A)) + Q - dot((beta * tri_dot(A.T, p0, B) + W), f0)
            f1 = l_div((R + beta * tri_dot(B.T, p1, B)), (beta * tri_dot(B.T, p1, A) + W.T))
            # f1 = la.solve((R + beta * tri_dot(B.T, p1, B)), (beta * tri_dot(B.T, p1, A) + W.T))
            
            dd = max(abs(f1 - f0))
            it, p0 = it + 1, p1
           
        if it >= maxit: 
            print 'WARNING: Iteration limit of 1000 reached in OLRP'
        
        return f1, p0



def kfilter(A,C,V1,V2,V12=None):
    """
    KFILTER can have arguments: (A,C,V1,V2) if there are no cross 
    products, V12=0. 
    KFILTER calculates the kalman gain, k, and the stationary 
    covariance matrix, s, using the Kalman filter for:
  
		x[t+1] = Ax[t] + Bu[t] + w1[t+1]
  
             y[t] = Cx[t] + Du[t] + w2[t]

           E [w1(t+1)] [w1(t+1)]' =  [V1   V12;         
             [ w2(t) ] [ w2(t) ]      V12' V2 ]

    where x is the mx1 vector of states, u is the nx1 vector of controls, 
    y is the px1 vector of observables, A is mxm, B is mxn, C is pxm, 
    V1 is mxm, V2 is pxp, V12 is mxp.
    """
    m = max(A.shape)
    rc, cc = C.shape
    
    if V12 == None: V12 = np.zeros((m, rc))
    
    if np.rank(V2) == rc:
        A = A - tri_dot(V12, inv(V2), C)
        V1 = V1 - tri_dot(V12, inv(V2), V12.T)
        k, s = doubleo(A, C, V1, V2)
               
        return k + dot(V12, inv(V2)), s
               
    else:
        s0 = .01 * np.eye(m)
        dd, it, maxit = 1, 1, 1000
        
        while dd > 1e-8 and it <= maxit:
            k0 = r_div((tri_dot(A, s0, C.T) + V12), (V2 + tri_dot(C, s0, C.T)))
            s1 = (tri_dot(A, s0, A.T)) + V1 - dot((tri_dot(A, s0, C.T) + V12), k0.T)
            k1 = r_div((tri_dot(A, s1, C.T) + V12), (V2 + tri_dot(C, s1, C.T)))
            dd = max(np.abs(k1 - k0))
            it, s0 = it+1, s1

        if it >= maxit:
            print 'WARNING: Iteration limit of 1000 reached in KFILTER.M'
        
        return k1, s0
