import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from .utility import comp_data_covariances, progprint_xrange
from .ssid_loss import f_l2_Hankel_nl
from .ssid_gradients import g_l2_Hankel_sgd_nl_rnd, g_l2_Hankel_sgd_ln_rnd
from .ssid_gradients import g_l2_Hankel_sgd_nl_sso, g_l2_Hankel_sgd_ln_sso
from .ssid_gradients import g_l2_Hankel_sgd_nl, g_l2_Hankel_sgd_ln

###########################################################################
# stochastic gradient descent: gradients w.r.t. C,R,A,B,X=cov(x_{t+m},x_t)
###########################################################################

def main(lag_range,n,y,obs_scheme,
         Qs=None, Om=None,
         parametrization='nl',
         sso=False, W=None, 
         idx_a=None,idx_b=None,
         pars_init='default', aux_init=None,
         alpha=0.001, b1=0.9, b2=0.99, e=1e-8, a_decay = 1., 
         max_iter=100, max_epoch_size=np.inf, eps_conv=0.99999,
         batch_size=1,save_every=np.inf,data_path=None,mmap=False, 
         return_aux=False, verbose=False, pars_track=None,
         dtype=np.float):
    
    """ Fits LDS in multiple stages, commonly seperated by different batch sizes.

    Parameters
    ----------
    lag_range : array
        array of increasing time-lags to consider
    n : int
        latent dimensionality 
    y : array
        T-by-p array of observed data (unobserved entries will not be queried)
    obs_scheme : ObservationScheme object
        block-wise partial observation scheme
    Qs : list of arrays
        (masked) pairwise time-lagged covariances. Will be computed if None
    Om : list of arrays
        masks for pairwise time-lagged covariances. Will be computed if None
    parametrization : str
        str giving parametrization of latent dynamics model .
        supported parametrizations: 'ln' linear, 'nl' agnostic
    sso : boolean
        if True, assumes block-wise observation scheme (faster computations)
    W :  list or arrays
        list of (inverse) co-observation counts. Will be computed if None
    idx_a, idx_b : arrays
        index arrays for which (variable pairs) to track pairwise covariances
    pars_init: str or dict
        initial parameters. 'default' for default initialization scheme
    aux_init: list of arrays or None
        optional ADAM auxiliary variables (m,v) for initialization of SGD
    alpha, b1, b2, e: floats
        ADAM parameters
    a_decay : float
        decay parameter of ADAM learning rate: a = alpha* a_decay**epoch
    max_iter : int
        maximum number of epochs
    max_epoch_size : int
        maxmimum epoch length 
    eps_conv: float
        convergence criterion. Currently not used with minibatch-SGD !
    batch_size : int 
        batch size for minibatch stochastic gradient descent 
    save_every: int
        counter for epochs, how often to save intermediate results to data_path
    mmap: boolean
        flag if y is stored as memory-mapped array (see numpy.memmap)
    return_aux: boolean
        optionally returns ADAM auxiliary variables (m,v) for later reuse      
    verbose : boolean 
        verbosity flag. Defines whether to print intermediate losses
    pars_track: None or function
        optional function to track quality of emission matrix C across epochs
    dtype : numpy datatype 
        datatype for numpy arrays

    Output
    ----------
    pars_est : dict
        parameter estimate
    pars_init : dict
        initial parameter estimate before first SGD step
    traces : list 
        list of fit diagnostics (loss & correlation of covariances over time)
    Qs : list of arrays
        (masked) pairwise time-lagged covariances. Same as input Qs if provided
    Om : list of arrays
        mask for pairwise time-lagged covariances. Same as input Om if provided
    W :  list or arrays
        list of (inverse) co-observation counts. Same as input W if provided
    t_descts : float
        total computation time
    """

    T, p = y.shape

    assert parametrization in ['nl', 'ln']    
    num_pars = 3 if parametrization=='nl' else 4

    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert np.all(idx_a == np.sort(idx_a))
    assert np.all(idx_a == np.sort(idx_a))

    if W is None:
        W = obs_scheme.comp_coocurrence_weights(lag_range=lag_range, 
                                                sso=sso,
                                                idx_a=idx_a,
                                                idx_b=idx_b) 

    if Qs is None or Om is None:
        Qs, Om = comp_data_covariances(n=n,y=y,lag_range=lag_range,sso=sso,
                                mmap=mmap,data_path=data_path,
                                obs_scheme=obs_scheme,
                                idx_a=idx_a,idx_b=idx_b,W=W)

    if not save_every == np.inf:
        assert not data_path is None

    alpha = alpha * np.ones(num_pars) if np.asarray(alpha).size==1 else alpha
    b1    =  b1   * np.ones(num_pars) if np.asarray(  b1 ).size==1 else b1
    b2    =  b2   * np.ones(num_pars) if np.asarray(  b2 ).size==1 else b2
    e     =   e   * np.ones(num_pars) if np.asarray(  e  ).size==1 else e

    T,p = y.shape 
    kl = len(lag_range)
    assert np.all(lag_range == np.sort(lag_range))

    if isinstance(pars_init, dict):
        assert 'C' in pars_init.keys()
        pars_init = pars_init.copy()

    if pars_init =='default':
        pars_init = {'A'  : np.diag(np.linspace(0.89, 0.91, n)),
             'Pi' : np.eye(n,dtype=dtype),
             'B'  : np.eye(n,dtype=dtype), 
             'C'  : np.random.normal(size=(p,n)).astype(dtype=dtype),
             'R'  : np.zeros(p,dtype=dtype),
             'X'  : np.zeros(((kl)*n, n),dtype=dtype)} 

    f,g,batch_draw,track_corrs,converged,save_interm = sis_setup(
                                           lag_range=lag_range,n=n,T=T,
                                           parametrization=parametrization,
                                           sso=sso,
                                           y=y,Qs=Qs,Om=Om,
                                           idx_a=idx_a, idx_b=idx_b,
                                           obs_scheme=obs_scheme, W=W,
                                           batch_size=batch_size,
                                           mmap=mmap, data_path=data_path,
                                           save_every=save_every, 
                                           max_iter=max_iter, eps_conv=eps_conv)
    if verbose:
        print('starting descent')    

    t_desc = time.time()
    pars_est, traces = adam_main(f=f,g=g,kl=kl,num_pars=num_pars,
                                 pars_init=pars_init,aux_init=aux_init, 
                                 alpha=alpha,b1=b1,b2=b2,e=e, a_decay=a_decay,
                                 batch_draw=batch_draw,converged=converged,
                                 track_corrs=track_corrs,save_interm=save_interm,
                                 max_iter=max_iter,max_epoch_size=max_epoch_size,
                                 batch_size=batch_size,return_aux=return_aux,
                                 verbose=verbose,pars_track=pars_track)

    t_desc = time.time() - t_desc

    if parametrization=='nl':

        pars_est = {'C': pars_est[0], 
                    'X': pars_est[1], 
                    'R': pars_est[2], 
                    'A': None, 
                    'Pi': pars_est[1][:n,:].copy(), 
                    'B': None }

    elif parametrization=='ln':

        X, Pi = np.zeros((len(lag_range)*n, n)), pars_est[2].dot(pars_est[2].T)
        for m in range(len(lag_range)):
            m_ = lag_range[m]
            X[m*n:(m+1)*n,:] = np.linalg.matrix_power(pars_est[1],m_).dot(Pi)   

        pars_est = {'C': pars_est[0], 
                    'A': pars_est[1], 
                    'B': pars_est[2], 
                    'R': pars_est[3], 
                    'Pi': Pi,
                    'X': X}

    return pars_est, pars_init, traces, Qs, Om, W, t_desc


# decorations

def sis_setup(lag_range,T,n,y,Qs,Om,obs_scheme,
              idx_a=None, idx_b=None, W=None,
              parametrization='nl', sso=False, 
              max_iter=np.inf, eps_conv=0.99999, save_every=np.inf,
              batch_size=None, mmap=False, data_path=None):
    "returns error function and gradient for use with gradient descent solvers"

    sub_pops= obs_scheme.sub_pops
    idx_grp = obs_scheme.idx_grp
    obs_idx = obs_scheme.obs_idx
    obs_pops= obs_scheme.obs_pops
    obs_time= obs_scheme.obs_time

    T,p = y.shape
    kl = len(lag_range)
    kl_ = np.max(lag_range)+1

    fn = 'p'+str(p)+'n'+str(n)+'T'+str(T)+'subpops'+str(len(sub_pops))

    if sso and obs_scheme.use_mask and len(idx_grp) == 1:
        #print('using rnd gradients')        
        g_l2_nl, g_l2_ln = g_l2_Hankel_sgd_nl_rnd, g_l2_Hankel_sgd_ln_rnd
    elif sso:
        #print('using sso gradients')
        g_l2_nl, g_l2_ln = g_l2_Hankel_sgd_nl_sso, g_l2_Hankel_sgd_ln_sso 
    else:
        g_l2_nl, g_l2_ln = g_l2_Hankel_sgd_nl, g_l2_Hankel_sgd_ln


    anb = np.intersect1d(idx_a, idx_b)
    idx_Rb = np.where(np.in1d(idx_b,idx_a))[0]
    idx_Ra = np.where(np.in1d(idx_a,idx_b))[0]

    if parametrization=='nl':

        def g(pars, ts, ms):
            return  g_l2_nl(C=pars[0],X=pars[1],R=pars[2],y=y,
                lag_range=lag_range,ts=ts,ms=ms,obs_scheme=obs_scheme,W=W)

        def f(pars):
            return f_l2_Hankel_nl(C=pars[0],X=pars[1],R=pars[2],Qs=Qs,Om=Om,
                lag_range=lag_range, ms=range(kl),idx_a=idx_a,idx_b=idx_b,
                anb=anb, idx_Ra=idx_Ra, idx_Rb=idx_Rb)

        def track_corrs(pars) :
            return track_correlations(C=pars[0],A=None,B=None,X=pars[1],
                            R=pars[2], Qs=Qs, p=p, n=n, lag_range=lag_range,
                            idx_a=idx_a, idx_b=idx_b, mmap=mmap, 
                            data_path=data_path)

        def save_interm(pars, t):

            if np.mod(t, save_every) == 0:

                try:
                    print('saving intermediate results...')
                    save_dict = {'T' : T,
                                 'lag_range' : lag_range,
                                 'C' : pars[0],
                                 'X' : pars[1],
                                 'R' : pars[2],
                                 'mmap' : mmap,
                                 'data_path' : data_path
                                }

                    np.savez(data_path +  fn + '_interm_i'+str(t), save_dict)

                except:
                    print('failed to save interm. results! continuing fit ...')

    elif parametrization=='ln':

        def g(pars, ts, ms):
            return  g_l2_ln(C=pars[0],A=pars[1],B=pars[2],R=pars[3],y=y, 
                lag_range=lag_range,ts=ts,ms=ms,obs_scheme=obs_scheme,W=W)

        def f(pars):
            X, Pi = np.zeros((kl_*n,n)), pars[2].dot(pars[2].T)
            for m in lag_range:
                X[m*n:(m+1)*n,:] = np.linalg.matrix_power(pars[1],m).dot(Pi)                
            return f_l2_Hankel_nl(C=pars[0],X=X,R=pars[3],Qs=Qs,Om=Om,
                lag_range=lag_range, ms=range(kl),idx_a=idx_a,idx_b=idx_b,
                anb=anb, idx_Ra=idx_Ra, idx_Rb=idx_Rb)

        def track_corrs(pars) :
            return track_correlations(C=pars[0],A=pars[1],B=pars[2],
                            X=None,R=pars[3],
                            Qs=Qs, p=p, n=n, lag_range=lag_range,
                            idx_a=idx_a, idx_b=idx_b, mmap=mmap, 
                            data_path=data_path)

        def save_interm(pars, t):

            if np.mod(t, save_every) == 0:

                try:
                    print('saving intermediate results...')
                    save_dict = {'T' : T,
                                 'lag_range' : lag_range,
                                 'C' : pars[0],
                                 'A' : pars[1],
                                 'B' : pars[2],
                                 'R' : pars[3],
                                 'mmap' : mmap,
                                 'data_path' : data_path
                                }

                    np.savez(data_path +  fn + '_interm_i'+str(t), save_dict)
                    
                except:
                    print('failed to save interm. results! continuing fit ...')

    # setting up the stochastic batch selection:
    batch_draw, g_sgd = sgd_draw(p, T, lag_range, batch_size, g)

    # set convergence criterion 
    if batch_size is None:
        def converged(t, fun):
            if t>= max_iter:
                return True
            elif t > 100 and fun[t-1] > eps_conv * np.min(fun[t-100:t-1]):
                return True
            else:
                return False

    else:
        def converged(t, fun):
            return True if t >= max_iter else False

    return f, g_sgd, batch_draw, track_corrs, converged, save_interm

def sgd_draw(p, T, lag_range, batch_size, g):
    "returns sequence of indices for sets of neuron pairs for SGD"

    kl = len(lag_range)
    kl_ = np.max(lag_range)+1
    if batch_size is None:

        def batch_draw():
            ts = (np.random.permutation(np.arange(T - kl_)) , )
            ms = (lag_range, )   
            return ts, ms
        def g_sgd(pars, ts, ms, i):
            return g(pars, ts[i], ms[i])

    elif batch_size == 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - kl_))
            ms = np.random.randint(0, kl, size=(len(ts),))         
            return ts, ms
        def g_sgd(pars, ts, ms, i):
            return g(pars, (ts[i],), (ms[i],))

    elif batch_size > 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0,kl,size=len(ts)//batch_size) 
            rnge = range(len(ts)//batch_size)
            return ([ts[i*batch_size:(i+1)*batch_size] for i in rnge],ms)

        def g_sgd(pars, ts, ms, i):
            return g(pars, ts[i], (ms[i],))

    return batch_draw, g_sgd


# main optimiser

def adam_main(f,g,pars_init,aux_init,num_pars,kl,alpha,b1,b2,e,a_decay,
            batch_draw,track_corrs,converged,save_interm,
            return_aux,max_iter,batch_size,max_epoch_size,
            verbose,pars_track,dtype=np.float):
    " main function for ADAM optimization "
    # initialise pars
    p, n = pars_init['C'].shape
    pars = init_adam(pars_init, p, n, kl, num_pars)

    # setting up Adam
    b1,b2,e,vp,mp = pars_adam(batch_size,p,n,kl,num_pars,b1,b2,e,dtype)
    t_iter, t, ct_iter = 0, 0, 0 
    corrs  = np.zeros((kl, 12))

    def epoch_range(epoch_size):
        if p > 1e4:
            return progprint_xrange(epoch_size, perline=100)
        else:
            return range(epoch_size)

    # trace function values
    fun = np.empty(max_iter)    

    corrs[:,ct_iter] = track_corrs(pars) 
    ct_iter += 1

    while not converged(t_iter, fun):

        ts, ms = batch_draw()        
        epoch_size = get_epoch_size(batch_size, p, ts, max_epoch_size)

        for idx_epoch in epoch_range(epoch_size):
            t += 1

            grads = g(pars, ts, ms, idx_epoch)
            pars, mp, vp = adam_step(pars,grads,mp,vp,alpha,b1,b2,e,t_iter+1)

        if t_iter < max_iter:          # really expensive!
            fun[t_iter] = f(pars)
            if not pars_track is None:
                pars_track(pars,t_iter)
        if verbose and p > 1e4:
            print('f = ', fun[t_iter])
        if verbose and np.mod(t_iter,max_iter//10) == 0:
            print('finished %', 100*t_iter/max_iter+10)
            print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(pars) 
            ct_iter += 1

        t_iter += 1
        alpha *= a_decay

        save_interm(pars, t_iter)

    corrs[:,ct_iter] = track_corrs(pars)
    fun = fun[:t_iter]

    if verbose:
        print('total iterations: ', t)

    traces = (fun,corrs,(mp,vp)) if return_aux else (fun,corrs)
    return pars, traces    

def pars_adam(batch_size,p,n,kl,num_pars,b1,b2,e,dtype=np.float):
    " returns initial ADAM parameters "

    if batch_size is None:
        print('using batch gradients - switching to plain gradient descent')
        b1 = np.zeros(num_pars,dtype=dtype)
        b2 = np.ones(num_pars, dtype=dtype) 
        e  = np.zeros(num_pars,dtype=dtype) 
    elif not batch_size >= 1: 
        raise Exception('cannot handle selected batch size')

    if num_pars == 3:     # (C, X, R)
        vp_0 = [np.ones((p,n),dtype=dtype),  
                np.ones((kl*n,n),dtype=dtype),  
                np.ones(p,dtype=dtype)]
        mp_0 = [np.zeros((p,n),dtype=dtype), 
                np.zeros((kl*n,n),dtype=dtype), 
                np.zeros(p,dtype=dtype)]
    elif num_pars == 4:   # (C, A, B, R)
        vp_0 = [np.ones((p,n),dtype=dtype),  
                np.ones((n,n),dtype=dtype),  
                np.ones((n,n),dtype=dtype),  
                np.ones(p,dtype=dtype)]
        mp_0 = [np.zeros((p,n),dtype=dtype), 
                np.zeros((n,n),dtype=dtype), 
                np.zeros((n,n),dtype=dtype), 
                np.zeros(p,dtype=dtype)]

    return b1,b2,e,vp_0,mp_0

def adam_step(pars,g,m,v,a,b1,b2,e,t):

    for i in range(len(m)):

        m[i] = (b1[i] * m[i] + (1-b1[i]) * g[i])
        v[i] = (b2[i] * v[i] + (1-b2[i]) * g[i]**2)

        mh,vh = adam_norm(m[i],v[i],b1[i],b2[i],t)

        pars[i] -= a[i] * mh / (np.sqrt(vh) + e[i])

    return pars, m, v

def adam_norm(m, v, b1, b2,t):

    m = m / (1-b1**t) if b1 != 1 else m.copy()
    v = v / (1-b2**t) if b2 != 1 else v.copy()

    return m,v

def init_adam(pars_0, p, n, kl, num_pars):
    " returns initial LDS parameters where not provided "

    dtype = pars_0['C'].dtype

    if num_pars == 3:
        C = pars_0['C'].copy()
        if 'R' in pars_0.keys():
            R = pars_0['R'].copy() 
        else:
            R = np.zeros(p,dtype=dtype)
        if 'X' in pars_0.keys():
            X = pars_0['X'].copy() 
        else:
            X = np.zeros((n*kl, n),dtype=dtype)
        pars = [C,X,R]

    elif num_pars == 4:
        C = pars_0['C'].copy()
        if 'A' in pars_0.keys():
            A = pars_0['A'].copy() 
        else:
            A = np.zeros((n, n),dtype=dtype)
        if 'B' in pars_0.keys():
            B = pars_0['B'].copy() 
        else:
            B = np.zeros((n, n),dtype=dtype)
        if 'R' in pars_0.keys():
            R = pars_0['R'].copy() 
        else:
            R = np.zeros(p,dtype=dtype)
        pars = [C,A,B,R]

    return pars

def get_epoch_size(batch_size, p=None, a=None, max_epoch_size=np.inf):

    if batch_size is None:
        epoch_size = len(a)
    elif batch_size >= 1:
        epoch_size = len(a)
    
    return int(np.min((epoch_size, max_epoch_size)))      

def track_correlations(C, A, B, X, R, Qs, p, n, lag_range,
    idx_a=None, idx_b=None, mmap = False, data_path=None):
    """ Computes correlation of covariances. 
        Uses all given covariance entries (no masking)!
    """
    Pi = None if B is None else B.dot(B.T) 

    if X is None:
        X = np.vstack([np.linalg.matrix_power(A,m) for m in lag_range]).dot(Pi)

    kl = len(lag_range)
    corrs = np.nan * np.ones(kl)
    if not Qs is None:

        idx_a = np.arange(p) if idx_a is None else idx_a
        idx_b = idx_a if idx_b is None else idx_b
        idx_ab = np.intersect1d(idx_a, idx_b)
        idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
        idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]

        assert (len(idx_a), len(idx_b)) == Qs[0].shape
        for m in range(kl):
            m_ = lag_range[m] 
            Qrec = C[idx_a,:].dot(X[m*n:(m+1)*n, :]).dot(C[idx_b,:].T) 
            if m_==0:
                Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(R[idx_ab])
            
            if mmap:
                Q = np.load(data_path+'Qs_'+str(m_)+'.npy')
            else:
                Q = Qs[m]
                
            corrs[m] = np.corrcoef( Qrec.reshape(-1), Q.reshape(-1) )[0,1]
            
            if mmap:
                del Q                

    return corrs
