import numpy as np

# evaluation of target loss functions

def f_blank(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b):

    return 0.

def f_l2_Hankel_nl(C,X,R,Qs,Om,lag_range,ms,idx_a,idx_b, anb=None, idx_Ra=None, idx_Rb=None):
    "Hankel reconstruction error for dynamical systems with general latent dynamics"

    p,n = C.shape
    L = 0.

    for m in ms:
        CXC = C[idx_a,:].dot(X[m*n:(m+1)*n,:]).dot(C[idx_b,:].T)
        if lag_range[m]==0:
            anb = np.intersect1d(idx_a,idx_b) if anb is None else anb
            if len(anb) > 0:
                idx_Rb = np.where(np.in1d(idx_b,idx_a))[0] if idx_Rb is None else idx_Rb
                idx_Ra = np.where(np.in1d(idx_a,idx_b))[0] if idx_Ra is None else idx_Ra
                CXC[idx_Ra, idx_Rb] += R[anb]
        L += np.sum( (CXC - Qs[m])[Om[m]]**2)

    return 0.5 * L

def f_l2_block(C,AmPi,Q,idx_grp,co_obs,idx_a,idx_b,W=None):
    "Hankel reconstruction error on an individual Hankel block"

    err = 0.
    for i in range(len(idx_grp)):
        err_ab = 0.
        a = np.intersect1d(idx_grp[i],idx_a)
        b = np.intersect1d(co_obs[i], idx_b)
        a_Q = np.in1d(idx_a, a)
        b_Q = np.in1d(idx_b, b)

        v = (C[a,:].dot(AmPi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
        v = v.reshape(-1,) if  W is None else W.reshape(-1,) * v.reshape(-1,)

        err += v.dot(v)

    return err

def f_l2_inst(C,Pi,R,Q,idx_grp,co_obs,idx_a,idx_b,W=None):
    "reconstruction error on the instantaneous covariance"

    err = 0.
    if not Q is None:
        for i in range(len(idx_grp)):

            a = np.intersect1d(idx_grp[i],idx_a)
            b = np.intersect1d(co_obs[i], idx_b)
            a_Q = np.in1d(idx_a, a)
            b_Q = np.in1d(idx_b, b)

            v = (C[a,:].dot(Pi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
            idx_R = np.where(np.in1d(b,a))[0]
            v[np.arange(len(idx_R)), idx_R] += R[a]
            v = v.reshape(-1,) if  W is None else W.reshape(-1,)*v.reshape(-1,)

            err += v.dot(v)

    return err
