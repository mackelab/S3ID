import numpy as np

# basic gradients 

# No exploitation of index group structure. Co-ocurrence weights W[m] are p-by-p. 
# Slow. 

def g_l2_Hankel_sgd_nl(C,X,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    grad_C = np.zeros_like(C)
    grad_X = np.zeros_like(X)
    grad_R = np.zeros_like(R)

    get_observed = obs_scheme.gen_get_observed()

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        grad_Xm = grad_X[m*n:(m+1)*n, :]
        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            g_C_l2_vector_pair_rw(grad_C,  m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rw(grad_Xm, m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])

        if m_==0:
            g_R_l2_Hankel_sgd_rw(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    return grad_C / len(ts), grad_X / len(ts), grad_R / len(ts)

def g_l2_Hankel_sgd_ln(C,A,B,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    kl_ = np.max(lag_range)+1 # d/dA often needs all powers A^m

    get_observed = obs_scheme.gen_get_observed()

    grad_C = np.zeros_like(C)
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_R = np.zeros_like(R)

    Pi = B.dot(B.T)
    Aexpm = np.zeros((kl_*n,n),dtype=A.dtype)
    Aexpm[:n,:] = np.eye(n,dtype=A.dtype)
    for m in range(1,kl_):
        Aexpm[m*n:(m+1)*n,:] = A.dot(Aexpm[(m-1)*n:(m)*n,:])
    grad_X = np.zeros_like(Aexpm)

    for m in ms:

        m_ = lag_range[m]
        Xm = Aexpm[m*n:(m+1)*n,:].dot(Pi)

        grad_Bm  = np.zeros_like(B)
        grad_Xm = grad_X[m*n:(m+1)*n, :]

        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            g_C_l2_vector_pair_rw(grad_C,  m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])
            g_B_l2_vector_pair_rw(grad_Bm, m_, C, Aexpm[m*n:(m+1)*n,:], Pi, R, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rw(grad_Xm, m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])

        grad_B += (grad_Bm + grad_Bm.T)

        g_A_l2_block(grad_A, grad_Xm.dot(Pi), Aexpm,m) 

        if m_==0:
            g_R_l2_Hankel_sgd_rw(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    grad_B = grad_B.dot(B)

    return grad_C/len(ts), grad_A/len(ts), grad_B/len(ts), grad_R/len(ts)

def g_A_l2_block(grad, dXmPi, Aexpm, m):
    "returns l2 Hankel reconstr. gradient w.r.t. A for a single Hankel block"

    n = Aexpm.shape[1]

    for q in range(m):
        grad += Aexpm[q*n:(q+1)*n,:].T.dot(dXmPi.dot(Aexpm[(m-1-q)*n:(m-q)*n,:].T))

def g_C_l2_vector_pair_rw(grad, m_, C, Xm, R, a, b, anb, yp, yf, Wm):

    C___ = C.dot(Xm)   # mad-
    C_tr = C.dot(Xm.T) # ness        
        
    for k in a:
        WC = C_tr[b,:] * Wm[k,b].reshape(-1,1)
        grad[k,:] += C[k,:].dot( C_tr[b,:].T.dot(WC) ) 
        grad[k,:] -= yf[k] * yp[b].dot(WC)
        
    for k in b:
        WC = C___[a,:] * Wm[a,k].reshape(-1,1)
        grad[k,:] += C[k,:].dot( C___[a,:].T.dot(WC) ) 
        grad[k,:] -= yp[k] * yf[a].dot(WC)
        
    if m_ == 0:      
        grad[anb,:] += (R[anb]*Wm[anb,anb]).reshape(-1,1)*(C___[anb,:] + C_tr[anb,:])

def g_B_l2_vector_pair_rw(grad, m_, C, Am, Pi, R, a, b, anb, yp, yf, Wm):

    for k in a:        
        CAm_k = C[k,:].dot(Am)
        S_k = C[b,:].T.dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad += np.outer(CAm_k, CAm_k).dot(Pi).dot(S_k)
        S_k = yp[b].dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad -= np.outer(yf[k] * CAm_k, S_k)

    if m_ == 0:
        grad += C[anb,:].dot(Am).T.dot( (R[anb] * Wm[anb,anb]).reshape(-1,1)*C[anb,:]) 

def g_X_l2_vector_pair_rw(grad, m_, C, Xm, R, a, b, anb, yp, yf, Wm):

    for k in a:        
        S_k = C[b,:].T.dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad += np.outer(C[k,:], C[k,:]).dot(Xm).dot(S_k)
        
        S_k = yp[b].dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad -= np.outer(yf[k] * C[k,:], S_k)
        
    if m_ == 0:
        grad += C[anb,:].T.dot( (R[anb] * Wm[anb,anb]).reshape(-1,1)*C[anb,:]) 

def g_R_l2_Hankel_sgd_rw(grad, C, X0, R, y, ts, get_observed, W0):

    for t in ts:
        b = get_observed(t)         
        grad[b] += (R[b] + np.sum(C[b,:] * C[b,:].dot(X0.T),axis=1) - y[t,b]**2) * W0[b,b]


# gradients for data missing at random

# Assuming co-occurence weights W[m][i,j] to be similar for all i,j (valid for large T). 
# Implementation assumes single index group (reads only W[m][0,0]). 

def g_l2_Hankel_sgd_nl_rnd(C,X,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    grad_C = np.zeros_like(C)
    grad_X = np.zeros_like(X)
    grad_R = np.zeros_like(R)

    get_observed = obs_scheme.gen_get_observed()

    CC = C.T.dot(C)

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        grad_Xm = grad_X[m*n:(m+1)*n, :]
        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            CC = C[b,:].T.dot(C[b,:])

            g_C_l2_vector_pair_rnd(grad_C,  m_, C, Xm, R, CC, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rnd(grad_Xm, m_, C, Xm, R, CC, a, b, anb, y[t], y[t+m_], W[m])

        if m_==0:
            g_R_l2_Hankel_sgd_rnd(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    return grad_C / len(ts), grad_X / len(ts), grad_R / len(ts)

def g_l2_Hankel_sgd_ln_rnd(C,A,B,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    kl_ = np.max(lag_range)+1 # d/dA often (but not always) needs all powers A^m

    get_observed = obs_scheme.gen_get_observed()

    grad_C = np.zeros_like(C)
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_R = np.zeros_like(R)

    Pi = B.dot(B.T)
    Aexpm = np.zeros((kl_*n,n))
    Aexpm[:n,:] = np.eye(n)
    for m in range(1,kl_):
        Aexpm[m*n:(m+1)*n,:] = A.dot(Aexpm[(m-1)*n:(m)*n,:])
    grad_X = np.zeros_like(Aexpm, dtype=A.dtype)

    for m in ms:

        m_ = lag_range[m]
        Xm = Aexpm[m*n:(m+1)*n,:].dot(Pi)

        grad_Bm  = np.zeros_like(B)
        grad_Xm = grad_X[m*n:(m+1)*n, :]

        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            CC = C[b,:].T.dot(C[b,:])

            g_C_l2_vector_pair_rnd(grad_C,  m_, C, Xm, R, CC, a, b, anb, y[t], y[t+m_], W[m])
            g_B_l2_vector_pair_rnd(grad_Bm, m_, C, Aexpm[m*n:(m+1)*n,:], Pi, R, CC, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rnd(grad_Xm, m_, C, Xm, R, CC, a, b, anb, y[t], y[t+m_], W[m])

        grad_B += (grad_Bm + grad_Bm.T)

        g_A_l2_block(grad_A, grad_Xm.dot(Pi), Aexpm,m) # grad_Xm.dot(Pi) possibly too costly

        if m_==0:
            g_R_l2_Hankel_sgd_rnd(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    grad_B = grad_B.dot(B)

    return grad_C / len(ts), grad_A / len(ts), grad_B / len(ts), grad_R / len(ts)


def g_C_l2_vector_pair_rnd(grad, m_, C, Xm, R, CC, a, b, anb, yp, yf, Wm):

    SC = CC * Wm[0,0]
    Sy = yp[b].dot(C[b,:]) * Wm[0,0]
    grad[a,:] += C[a,:].dot( Xm.dot(SC).dot(Xm.T) ) - np.outer(yf[a], Sy.dot(Xm.T))

    Sy = yf[a].dot(C[a,:]) * Wm[0,0]
    grad[b,:] += C[b,:].dot( Xm.T.dot(SC).dot(Xm) ) - np.outer(yp[b], Sy.dot(Xm))
        
    if m_ == 0:
        grad[anb,:] += (R[anb]*Wm[0,0]).reshape(-1,1) * (C[anb,:].dot(Xm+Xm.T))
           
def g_B_l2_vector_pair_rnd(grad, m_, C, Am, Pi, R, CC, a, b, anb, yp, yf, Wm):

    p,n = C.shape
    SC = CC * Wm[0,0]
    Sy = yp[b].dot(C[b,:] * Wm[0,0])
    grad += Am.T.dot(CC).dot(Am).dot(Pi).dot(SC)-np.outer(yf[a].dot(C[a,:]).dot(Am),Sy)

    if m_ == 0:
        grad += C[anb,:].dot(Am).T.dot( (R[anb] * Wm[0,0]).reshape(-1,1) * C[anb,:] )

def g_X_l2_vector_pair_rnd(grad, m_, C, Xm, R, CC, a, b, anb, yp, yf, Wm):

    p,n = C.shape
    SC = CC * Wm[0,0]
    Sy = yp[b].dot(C[b,:]) * Wm[0,0]
    grad += CC.dot(Xm).dot(SC)-np.outer(yf[a].dot(C[a,:]),Sy)
        
    if m_ == 0:
        grad += C[anb,:].T.dot( (R[anb] * Wm[0,0]).reshape(-1,1) * C[anb,:] )

def g_R_l2_Hankel_sgd_rnd(grad, C, X0, R, y, ts, get_observed, W0):

    for t in ts:
        b = get_observed(t)         
        grad[b] += (R[b] + np.sum(C[b,:] * C[b,:].dot(X0.T),axis=1) - y[t,b]**2) * W0[0,0]



# gradients for block-structured observations schemes (multiple partial observations)

# Full exploitation of index group structure, allowing vectorization of most computations.
# Co-occurence weights W[m] are len(idx_grp)-by-len(idx_grp) ( << p-by-p in most cases)

def g_l2_Hankel_sgd_ln_sso(C,A,B,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    kl_ = np.max(lag_range)+1 # d/dA often (but not always) needs all powers A^m

    get_idx_grp,idx_grp = obs_scheme.gen_get_idx_grp(), obs_scheme.idx_grp

    grad_C = np.zeros_like(C)
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_R = np.zeros_like(R)

    Pi = B.dot(B.T)

    # pre-compute
    inst_is_ = np.unique(np.hstack(get_idx_grp(ts)))
    all_is_  = np.union1d(np.unique([np.hstack(get_idx_grp(ts+m)) for m in ms]), inst_is_)
    CCs, R_CX0Cs = [], []
    for i in range(len(idx_grp)):
        CCs.append( C[idx_grp[i],:].T.dot(C[idx_grp[i],:]) if i in all_is_ else None) 
        R_CX0Cs.append(R[idx_grp[i]] + np.sum(C[idx_grp[i],:] * C[idx_grp[i],:].dot(Pi),axis=1) if i in inst_is_ else None)

    Aexpm = np.zeros((kl_*n,n), dtype=A.dtype)
    Aexpm[:n,:] = np.eye(n)
    for m in range(1,kl_):
        Aexpm[m*n:(m+1)*n,:] = A.dot(Aexpm[(m-1)*n:(m)*n,:])
    grad_X = np.zeros_like(Aexpm)

    for m in ms:

        m_ = lag_range[m]
        Xm = Aexpm[m*n:(m+1)*n,:].dot(Pi)

        grad_Bm = np.zeros_like(B)
        grad_Xm = grad_X[m*n:(m+1)*n, :]

        for t in ts:
            is_ = get_idx_grp(t+m_)
            js_ = get_idx_grp(t)
            inj = np.intersect1d(is_, js_)

            g_C_l2_vector_pair_sso(grad_C,  m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_B_l2_vector_pair_sso(grad_Bm, m_, C, Aexpm[m*n:(m+1)*n,:], Pi, R, CCs, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_sso(grad_Xm, m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            if m_ == 0:
                g_R_l2_Hankel_sgd_sso(grad_R, C, R_CX0Cs, idx_grp, inj, y[t], W[m])

        grad_B += (grad_Bm + grad_Bm.T)

        g_A_l2_block(grad_A, grad_Xm.dot(Pi), Aexpm,m) # grad_Xm.dot(Pi) possibly too costly

    grad_B = grad_B.dot(B)

    return grad_C / len(ts), grad_A / len(ts), grad_B / len(ts), grad_R / len(ts)


def g_l2_Hankel_sgd_nl_sso(C,X,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    grad_C = np.zeros_like(C)
    grad_X = np.zeros_like(X)
    grad_R = np.zeros_like(R)

    get_idx_grp,idx_grp = obs_scheme.gen_get_idx_grp(), obs_scheme.idx_grp

    # pre-compute
    inst_is_ = np.unique(np.hstack(get_idx_grp(ts)))
    all_is_  = np.union1d(np.unique([np.hstack(get_idx_grp(ts+m)) for m in ms]), inst_is_)
    CCs, R_CX0Cs = [], []
    for i in range(len(idx_grp)):
        CCs.append( C[idx_grp[i],:].T.dot(C[idx_grp[i],:]) if i in all_is_ else None) 
        R_CX0Cs.append(R[idx_grp[i]] + np.sum(C[idx_grp[i],:] * C[idx_grp[i],:].dot(X[:n,:].T),axis=1) if i in inst_is_ else None)

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        grad_Xm = grad_X[m*n:(m+1)*n, :]
        for t in ts:
            is_ = get_idx_grp(t+m_)
            js_ = get_idx_grp(t)
            inj = np.intersect1d(is_, js_)

            g_C_l2_vector_pair_sso(grad_C,  m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_sso(grad_Xm, m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])

            if m_ == 0:
                g_R_l2_Hankel_sgd_sso(grad_R, C, R_CX0Cs, idx_grp, inj, y[t], W[m])

    return grad_C / len(ts), grad_X / len(ts), grad_R / len(ts)

def g_C_l2_vector_pair_sso(grad, m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape

    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]        
            SC += CCs[j]* Wm[i,j]
            Sy += yp[b].dot(C[b,:]) * Wm[i,j]
        grad[a,:] += C[a,:].dot( Xm.dot(SC).dot(Xm.T) ) - np.outer(yf[a], Sy.dot(Xm.T))

    for j in js_:
        b = idx_grp[j]        
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for i in is_:        
            a = idx_grp[i]        
            SC += CCs[i] * Wm[i,j]
            Sy += yf[a].dot(C[a,:]) * Wm[i,j]
        grad[b,:] += C[b,:].dot( Xm.T.dot(SC).dot(Xm) ) - np.outer(yp[b], Sy.dot(Xm))
        
    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad[anb,:] += (R[anb]*Wm[i,i]).reshape(-1,1) * (C[anb,:].dot(Xm+Xm.T))

def g_B_l2_vector_pair_sso(grad, m_, C, Am, Pi, R, CCs, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape
    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros_like(Pi), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]
            SC += CCs[j] * Wm[i,j]
            Sy += yp[b].dot(C[b,:] * Wm[i,j])
        grad += Am.T.dot(CCs[i]).dot(Am).dot(Pi).dot(SC)-np.outer(yf[a].dot(C[a,:]).dot(Am),Sy)
    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad += C[anb,:].dot(Am).T.dot( (R[anb] * Wm[i,i]).reshape(-1,1) * C[anb,:] )

def g_X_l2_vector_pair_sso(grad, m_, C, Xm, R, CCs, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape
    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]
            SC += CCs[j] * Wm[i,j]
            Sy += yp[b].dot(C[b,:]) * Wm[i,j]
        grad += CCs[i].dot(Xm).dot(SC)-np.outer(yf[a].dot(C[a,:]),Sy)
        
    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad += C[anb,:].T.dot( (R[anb] * Wm[i,i]).reshape(-1,1) * C[anb,:] )

def g_R_l2_Hankel_sgd_sso(grad, C, R_CX0Cs, idx_grp, inj, yp, W0):

    for i in inj:
        b = idx_grp[i]         
        grad[b] += (R_CX0Cs[i] - yp[b]**2) * W0[i,i]