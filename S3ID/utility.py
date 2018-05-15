import numpy as np
from numpy.lib.stride_tricks import as_strided

import scipy as sp
from scipy import stats
from scipy import linalg as la

import matplotlib.pyplot as plt
import sys, time

###########################################################################
# Utility (general)
###########################################################################

def symmetrize(A):
    """Symmetrize matrix"""    
    return (A + A.T)/2


def blockarray(*args,**kwargs):
    """Block array from list of arrays"""    
    "Taken from Matthew J. Johnson's 'pybasicbayes' code package"
    return np.array(np.bmat(*args,**kwargs),copy=False)

def principal_angle(A, B):
    """ Computes principal angle between column-spaced of two matrices. 	
	Inputs A and B must be column-orthogonal."""    
    A = np.atleast_2d(A).T if (A.ndim<2) else A
    B = np.atleast_2d(B).T if (B.ndim<2) else B
    A = la.orth(A)
    B = la.orth(B)
    svd = la.svd(A.T.dot(B))[1]
    return np.arccos(np.minimum(svd, 1.)) / (np.pi/2.)


###########################################################################
# Utility (SSID-specific)
###########################################################################

def observability_mat(pars, k):
    """Compute analytic observability matrix for given set of LDS parameters

    Parameters
    ----------
    pars : dict or tuple
        parameters A, C for linear dynamical system
    k : int
        order parameter
    """
    if isinstance(pars, dict):
        A, C = pars['A'], pars['C']
    elif isinstance(pars, tuple) and len(pars) == 2:
        A, C = pars[0], pars[1]

    if len(C.shape)<2:
        C = C.reshape(1,C.size).copy()
    
    return blockarray([[C.dot(np.linalg.matrix_power(A,i))] for i in range(k)])


def reachability_mat(pars, l):
    """Compute analytic reachability matrix for given set of LDS parameters

    Parameters
    ----------
    pars : dict or tuple
        parameters A, B for linear dynamical system
    l : int
        order parameter
    """

    if isinstance(pars, dict):
        A, B = pars['A'], pars['B']
    elif isinstance(pars, tuple) and len(pars) == 2:
        A, B = pars[0], pars[1]  

    assert len(B.shape)>1

    return blockarray([np.linalg.matrix_power(A,i).dot(B) for i in range(l)])


def comp_model_covariances(pars, lag_range=None, mmap=False, chunksize=None, 
    data_path='../fits/', verbose=False):
    """" returns list of time-lagged covariances 
    cov(y_t+m, y_t) for  m = 1, ..., k+l-1"

    Parameters
    ----------
    pars : dict or tuple
        parameters A, C, Pi for linear dynamical system
    lag_range : array or list 
        array of time-lags. Can be non-contiguous. 
    mmap: boolean
        whether or not to use memory maps for result storage
    data_path: str
        path for result storage in case mmap = True
    chunksize: None or int
        int value will subdivide cov matrix computation into chunks 
        (for large system sizes p when memory is limited)
    verbose: boolean
        boolean for verbosity (relevant only for mmap=True)
    """
    
    lag_range = [] if lag_range is None else lag_range
    kl = range(len(lag_range))

    p,n = pars['C'].shape
    
    chunksize = p if chunksize is None else chunksize

    max_i = p//chunksize
    assert np.allclose(max_i * chunksize, p) 
        
    Qs = []
    for m in kl:
        m_ = lag_range[m]
        if mmap:
            Qs.append(np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, 
                mode='w+', shape=(p,p)))
            if verbose: 
                print('computing time-lagged covariance for lag m =', str(m))      
        else:
            Qs.append(np.empty((p,p)))

        if 'X' in pars.keys():
            APi = pars['X'][m*n:(m+1)*n,:]
        else:
            APi = np.linalg.matrix_power(pars['A'], m_).dot(pars['Pi'])
        for i in range(max_i):
            idx_i  = range(i*chunksize, (i+1)*chunksize)
            for j in range(max_i):
                idx_j = range(j*chunksize, (j+1)*chunksize)
                Qs[m][np.ix_(idx_i,idx_j)] = pars['C'][idx_i,:].dot( \
                    APi ).dot(pars['C'][idx_j,:].T) 
                if mmap:
                    del Qs[m]
                    Qs.append(np.memmap(data_path+'Qs_'+str(m_), 
                        dtype=np.float, mode='r+', shape=(p,p)))
        if mmap:
            del Qs[m]
            Qs.append(np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, 
                mode='r', shape=(p,p)))

        if lag_range[m] == 0:
            Qs[m][range(p),range(p)] += pars['R']

    return Qs


def comp_data_covariances(n,y,lag_range,obs_scheme,idx_a,idx_b,W,sso=False,
                          mmap=False,data_path=None,ts=None,ms=None):
    """" returns list of time-lagged covariances 
    cov(y_t+m, y_t) for  m = 1, ..., k+l-1"

    Parameters
    ----------
    n : int
        latent dimensionality
    y : array
        T-by-p array of observed activity 
    lag_range : array or list 
        array of time-lags for time-lagged covariances. Can be non-contiguous.
    idx_a, idx_b : arrays
        index arrays for pairwise time-lagged covariances. Subsets of range(p).
    W : list of arrays 
        co-ocurrence weight matrices, one per time-lag
    sso : boolean 
        if True, using multiple partial observation scheme
    mmap : boolean
        whether or not to use memory maps for result storage
    data_path : str
        path for result storage in case mmap = True
    ts : array or None
        indices for time points to use from y. Subset of range(T)
    ms : array or None
        indices of time-lags to use. Subset of range(len(lag_range))

    Output
    ----------
    Qs : list of arrays
        time-lagged covariance matrices
    Om : list of arrays
        boolean masks for co-observed variables (one per time-lag)    
    """
    T,p = y.shape
    kl_ = np.max(lag_range)+1
    pa, pb = len(idx_a), len(idx_b)
    idx_grp = obs_scheme.idx_grp

    ts = range(T-kl_) if ts is None else ts
    ms = range(len(lag_range)) if ms is None else ms

    Qs = [np.zeros((pa,pb), dtype=y.dtype) for m in range(len(lag_range))]
    Om = [np.zeros((pa,pb), dtype=bool) for m in range(len(lag_range))]

    if sso: 
        get_obs_idx = obs_scheme.gen_get_idx_grp()
        get_coobs_intervals = obs_scheme.gen_get_coobs_intervals(lag_range)
        idx_grp = obs_scheme.idx_grp
        for j in range(len(idx_grp)):
            b = np.intersect1d(idx_grp[j], idx_b)
            b_Q = np.in1d(idx_b, b)
            for i in range(len(idx_grp)):
                a = np.intersect1d(idx_grp[i], idx_a)
                a_Q = np.in1d(idx_a, a)
                for m in ms:
                    idx_coobs_ijm = get_coobs_intervals(j,i,m) # note ordering of j,i
                    if len(idx_coobs_ijm) > 0:                    
                        Qs[m][np.ix_(a_Q,b_Q)] = y[np.ix_(idx_coobs_ijm,a)].T.dot(y[np.ix_(idx_coobs_ijm-m,b)])
                        Om[m][np.ix_(a_Q,b_Q)] = True
    else:
        get_observed = obs_scheme.gen_get_observed()
        for m in ms:
            m_ = lag_range[m]
            for t in ts:
                a = np.intersect1d(get_observed(t+m_), idx_a)
                b = np.intersect1d(get_observed(t),    idx_b)
                a_Q = np.in1d(idx_a, a)
                b_Q = np.in1d(idx_b, b)

                Qs[m][np.ix_(a_Q, b_Q)] += np.outer(y[t+m_,a], y[t,b])
                Om[m][np.ix_(a_Q, b_Q)] = True

    if np.all(W[0].shape == (len(idx_grp), len(idx_grp))):
        for m in ms:
            for i in range(len(idx_grp)):
                for j in range(len(idx_grp)):

                    a = np.in1d(idx_a, np.intersect1d(idx_grp[i], idx_a))
                    b = np.in1d(idx_b, np.intersect1d(idx_grp[j], idx_b))

                    Qs[m][np.ix_(a, b)] *= W[m][i,j]

    elif np.all(W[0].shape == (p, p)):
        for m in ms:
            Qs[m] = Qs[m] * W[m][np.ix_(idx_a,idx_b)]

    else:
        raise Exception('shape misfit for weights W[m] at time-lag m=0')

    if mmap: # probably computing the Qs is costly
        for m in range(len(lag_range)):
            np.save(data_path+'Qs_'+str(lag_range[m]), Qs[m])

    return Qs, Om


def gen_data(p, n, lag_range, T, 
             nr=0 ,eig_m_r=0.98, eig_M_r=0.99, eig_m_c=0.98, eig_M_c=0.99, 
             mmap=False, chunksize=None, data_path='', idx_a=None, idx_b=None, 
             snr=(.75, 1.25), verbose=False, whiten=False, dtype=np.float):
    """" generate system parameters and toy data for linera dynamical system 

    Parameters
    ----------
    p : int
        observed system size
    n : int
        latent dimensionality
    lag_range : array or list 
        array of time-lags for time-lagged covariances. Can be non-contiguous.
    T : int
        length in bins of generated data trace
    nr : int
        number of real eigenvectors for linear latent dynamics. 0 < nr < n
    eig_* : float
        smallest (m) and largest (M) values for real (r) and complex (c) EVs
    mmap : boolean
        whether or not to use memory maps for result storage
    data_path : str
        path for result storage in case mmap = True
    chunksize : None or int
        int value will subdivide cov matrix computation into chunks 
        (for large system sizes p when memory is limited)
    idx_a, idx_b : arrays
        index arrays for pairwise time-lagged covariances. Subsets of range(p)
    snr : array or tuple
        signal-to-noise ratios for generated data (controls obseration noise R)
    verbose: boolean
        boolean for verbosity (relevant only for mmap=True)
    whiten: boolean
        if True, normalizes observed variances
    dtype: numpy dtype
        data type for output arrays   

    Output
    ----------
    pars_true : dict
        dictionary of LDS parameters used to generate the data
    x : array
        T-by-n array of generated latent activity 
    y : array
        T-by-p array of generated observed activity 
    idx_a, idx_b: arrays
        index arrays for pairwise time-lagged covariances.   
    """
    nr = n if nr is None else nr
    nc, nc_u = n - nr, (n - nr)//2
    assert nc_u * 2 == nc 

    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert np.all(idx_a == np.sort(idx_a))
    assert np.all(idx_b == np.sort(idx_b))
    pa, pb = len(idx_a), len(idx_b)

    ev_r = np.linspace(eig_m_r, eig_M_r, nr)
    ev_c = np.exp(2 * 1j * np.pi * np.random.vonmises(mu=0, kappa=1000, size=nc_u))
    ev_c = np.linspace(eig_m_c, eig_M_c, (n - nr)//2) * ev_c

    pars_true,Qs = gen_sys(p=p,n=n,lag_range=lag_range, nr=nr,ev_r=ev_r,ev_c=ev_c,
                           snr=snr, calc_stats=T==np.inf,
                           mmap=mmap,chunksize=chunksize,data_path=data_path,
                           whiten=whiten,dtype=dtype)
    pars_true['d'], pars_true['mu0'] = np.zeros(p,dtype=dtype), np.zeros(n,dtype=dtype), 
    pars_true['V0'] = pars_true['Pi'].copy()

    if T == np.inf:
        x,y = None, None
    else:
        x,y = draw_data(pars=pars_true, T=T, 
                        mmap=mmap, chunksize=chunksize, data_path=data_path,
                        dtype=dtype)

    return pars_true, x, y, idx_a, idx_b


def gen_sys(p, n, lag_range=None, nr=None, ev_r=None, ev_c=None,
            snr=(.75, 1.25), calc_stats=True, whiten=False,
            mmap=False, chunksize=None, data_path='../fits',dtype=np.float32):
    """" generate system parameters and covariances for linera dynamical system 

    Parameters
    ----------
    p : int
        observed system size
    n : int
        latent dimensionality
    lag_range : array or list 
        array of time-lags for time-lagged covariances. Can be non-contiguous.
    T : int
        length in bins of generated data trace
    nr : int
        number of real eigenvectors for linear latent dynamics. 0 < nr < n
    ev_r, ev_c : arrays
        real and complex eigenvalues
    snr : array or tuple
        signal-to-noise ratios for generated data (controls obseration noise R)
    calc_stats : boolean
        if True, computes pairwise time-lagged covariances
    whiten:
        if True, normalizes observed variances
    mmap : boolean
        whether or not to use memory maps for result storage
    data_path : str
        path for result storage in case mmap = True
    chunksize : None or int
        int value will subdivide cov matrix computation into chunks 
        (for large system sizes p when memory is limited)
    dtype: numpy dtype
        data type for output arrays   

    Output
    ----------
    pars_true : dict
        dictionary of LDS parameters used to generate the data
    Qs : list
        lists of pairwise time-lagged covariance matrices
    """
    kl = len(lag_range)
    pars_true = gen_pars(p,n,nr=nr,ev_r=ev_r,ev_c=ev_c,snr=snr,
                         whiten=whiten,dtype=dtype)
    if calc_stats:
        Qs = comp_model_covariances(pars_true, lag_range, mmap=mmap, 
            chunksize=chunksize,data_path=data_path)
    else:
        Qs = []
        for m in range(kl):
            Qs.append(None)

    return pars_true, Qs


def gen_pars(p,n, nr=None, ev_r = None, ev_c = None, 
             snr = (.75, 1.25), whiten=False,dtype=np.float32):
    """" generate system parameters for linera dynamical system 

    Parameters
    ----------
    p : int
        observed system size
    n : int
        latent dimensionality
    nr : int
        number of real eigenvectors for linear latent dynamics. 0 < nr < n
    ev_r, ev_c : arrays
        real and complex eigenvalues
    whiten: boolean
        if True, normalizes observed variances
    dtype: numpy dtype
        data type for output arrays   

    Output
    ----------
    pars_true : dict
        dictionary of LDS parameters used to generate the data
    """    
    nr = n if nr is None else nr
    nc, nc_u = n - nr, (n - nr)//2
    assert nc_u * 2 == nc 

    if not ev_r is None:
        assert ev_r.size == nr

    if not ev_c is None:
        assert ev_c.size == nc_u

    # generate dynamics matrix A
    Q, D = np.zeros((n,n), dtype=complex), np.zeros(n, dtype=complex)

    # draw real eigenvalues and eigenvectors
    D[:nr] = np.linspace(0.8, 0.99, nr) if ev_r is None else ev_r 
    Q[:,:nr] = np.random.normal(size=(n,nr))
    Q[:,:nr] /= np.sqrt((Q[:,:nr]**2).sum(axis=0)).reshape(1,nr)

    # draw complex eigenvalues and eigenvectors
    if ev_c is None:
        circs = np.exp(2 * 1j * np.pi * np.random.vonmises(mu=0, 
            kappa=1000, size=nc_u))
        scales = np.linspace(0.5, 0.9, nc_u)
        ev_c_r, ev_c_c = scales * np.real(circs), scales * np.imag(circs)
    else:
        ev_c_r, ev_c_c = np.real(ev_c), np.imag(ev_c) 
    V = np.random.normal(size=(n,n))
    for i in range(nc_u):
        Vi = V[:,i*2:(i+1)*2] / np.sqrt( np.sum(V[:,i*2:(i+1)*2]**2) )
        Q[:,nr+i], Q[:,nr+nc_u+i] = Vi[:,0]+1j*Vi[:,1], Vi[:,0]-1j*Vi[:,1] 
        D[nr+i], D[nr+i+nc_u] = ev_c_r[i]+1j*ev_c_c[i], ev_c_r[i]-1j*ev_c_c[i]

    A = Q.dot(np.diag(D)).dot(np.linalg.inv(Q))
    assert np.allclose(A, np.real(A))
    A = np.real(A)

    # generate innovation noise covariance matrix Q

    Q = np.atleast_2d(stats.wishart(5*n, np.eye(n)).rvs()/n)
    Pi = np.atleast_2d(sp.linalg.solve_discrete_lyapunov(A, Q))

    L = np.linalg.cholesky(Pi)
    Linv = np.linalg.inv(L)
    A, Q = Linv.dot(A).dot(L), Linv.dot(Q).dot(Linv.T)
    Pi = np.atleast_2d(sp.linalg.solve_discrete_lyapunov(A, Q))

    # generate emission-related matrices C, R

    C = np.random.normal(size=(p,n)) / np.sqrt(n)
    NSR = np.random.uniform(size=p, low=snr[0], high=snr[1]) # 1/SNR
    if whiten:
        C /= np.atleast_2d(np.sqrt(np.sum(C*C.dot(Pi), axis=1) * (1 + NSR))).T

    R = np.sum(C*C.dot(Pi), axis=1) * NSR

    try:
        B = np.linalg.cholesky(Pi)
    except:
        B = np.nan * np.ones((n,n))
        
    return { 'A': np.asarray(A,dtype=dtype), 
             'B': np.asarray(B,dtype=dtype), 
             'Q': np.asarray(Q,dtype=dtype), 
             'Pi': np.asarray(Pi,dtype=dtype), 
             'C': np.asarray(C,dtype=dtype), 
             'R': np.asarray(R,dtype=dtype) }


def draw_data(pars, T, mmap=False, chunksize=None, 
              data_path='../fits/', dtype=np.float):
    """" generate data for linera dynamical system 

    Parameters
    ----------
    pars : dict
        dictionary of LDS parameters used to generate the data    
    T : int
        length in bins of generated data trace
    mmap : boolean
        whether or not to use memory maps for result storage
    data_path : str
        path for result storage in case mmap = True
    chunksize : None or int
        int value will subdivide cov matrix computation into chunks 
        (for large system sizes p when memory is limited)
    dtype: numpy dtype
        data type for output arrays   

    Output
    ----------
    x : array
        T-by-n array of generated latent activity 
    y : array
        T-by-p array of generated observed activity     
    """   
    p,n = pars['C'].shape
    chunksize = p if chunksize is None else chunksize
    max_i = int(np.ceil(p/chunksize))

    def chunk_range(max_i):
        if p > 1000:
            return progprint_xrange(max_i, perline=100)
        else:
            return range(max_i)

    # start with noise terms
    L = np.linalg.cholesky(pars['Q'])
    x = np.asarray(np.random.normal(size=(T,n)), dtype=dtype)
    x = x.dot(L.T)

    # step thourhg latent dynamics
    x[0,:]  = pars['mu0'].copy() 
    x[0,:] += np.linalg.cholesky(pars['V0']).dot(np.random.normal(size=n))
    for t in range(1,T):
        x[t,:] += pars['A'].dot(x[t-1,:])

    # do emissions
    L = np.sqrt(pars['R'])
    if mmap:
        y = np.memmap(data_path+'y', dtype=dtype, mode='w+', shape=(T,p))
    else:
        y = np.empty(shape=(T,p),dtype=dtype)

    for i in chunk_range(max_i):
        idx_i = range(i*chunksize, np.minimum((i+1)*chunksize, p)) 
        y[:,idx_i] = np.asarray(np.random.normal(size=(T, len(idx_i))),
            dtype=dtype)*np.atleast_2d(L[idx_i]) \
                    + x.dot(pars['C'][idx_i,:].T)
        if mmap:
            del y # releases RAM, forces flush to disk
            y = np.memmap(data_path+'y', dtype=dtype, mode='r+', shape=(T,p))
    if mmap:
        del y # releases RAM, forces flush to disk
        y = np.memmap(data_path+'y', dtype=dtype, mode='r', shape=(T,p))

    return x, y


###########################################################################
# Utility (stitching-specific)
###########################################################################

def get_subpop_stats(sub_pops, p, obs_pops=None, verbose=False):
    """ Computes collection of helpful index sets for the stitching context

    Parameters
    ----------
    sub_pops : list or tuple of arrays
        index arrays for subpopulations (one array per subpopulation)
    p : int
        observed dimensionality (= total system size)
    obs_pops : tuple or None
        sequence of subpopulation indices giving their order of observation
    verbose: boolean
        verbosity flag

    Output
    ----------
    obs_idx : list of arrays
        index groups observed at each given time interval        
    idx_grp : list of arrays
        arrays of index group memberships, one per index group  
    co_obs : list
        list of index group lists, one per index group. Gives co-observations.
    overlaps: list of arrays
        list of overlaps between subpopulations
    overlap_grp : list of arrays 
        list of index groups found in more than one subpopulation
    idx_overlap : list of arrays 
        list of subpops in which the corresponding index group is found
    Om : p-by-p array or None
        mask for co-observed variables (instantaneous co-observation)
    Ovw : p-by-p array or None
        mask for overlaps across subpopulations 
    Ovc : p-by-p array or None
        mask for cross-overlaps (overlap across subpopualtions only) 
    """
    if obs_pops is None:
        obs_pops = tuple(range(len(sub_pops)))
    obs_idx, idx_grp = get_obs_index_groups(obs_scheme={'sub_pops': sub_pops,
        'obs_pops': obs_pops},p=p)
    overlaps, overlap_grp, idx_overlap = get_obs_index_overlaps(idx_grp, \
        sub_pops)

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    num_idx_grps, co_obs = len(idx_grp), []
    for i in range(num_idx_grps):    
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))

    if verbose:
        print('idx_grp:', idx_grp)
        print('obs_idx:', obs_idx)
        
    Om, Ovw, Ovc = comp_subpop_index_mats(sub_pops,idx_grp,overlap_grp,\
        idx_overlap)    
    
    if verbose:
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(Om,interpolation='none')
        plt.title('Observation pattern')
        plt.subplot(1,3,2)
        plt.imshow(Ovw,interpolation='none')
        plt.title('Overlap pattern')
        plt.subplot(1,3,3)
        plt.imshow(Ovc,interpolation='none')
        plt.title('Cross-overlap pattern')
        plt.show()
        
    return obs_idx,idx_grp,co_obs,overlaps,overlap_grp,idx_overlap,Om,Ovw,Ovc


def get_obs_index_groups(obs_scheme,p):
    """"  Computes index (or 'fate') groups for given observation scheme. 

    Index or 'fate' groups are non-overlapping groups of observed variables
    that are always jointly observed (or jointly unobserved), allowing to
    treat them jointly within a multiple partial observation scheme. 

    Parameters
    ----------
    obs_scheme: dict
        observation scheme with keys 'sub_pops', 'obs_time', 'obs_pops'
    p: int
        dimensionality of observed variables y

    Output
    ----------
    obs_idx : list of arrays
        index groups observed at each given time interval        
    idx_grp : list of arrays
        arrays of index group memberships, one per index group    
    """       
    try:
        sub_pops = obs_scheme['sub_pops'];
        obs_pops = obs_scheme['obs_pops'];
    except:
        raise Exception(('provided obs_scheme dictionary does not have '
                         'the required fields sub_pops and obs_pops.'))        

    J = np.zeros((p, len(sub_pops))) # binary matrix, each row gives which 
    for i in range(len(sub_pops)):      # subpopulations the observed variable
        if sub_pops[i].size > 0:        # y_i is part of
            J[sub_pops[i],i] = 1   

    twoexp = np.power(2,np.arange(len(sub_pops))) # we encode the binary rows 
    hsh = np.sum(J*twoexp,1)                     # of J using binary numbers

    lbls = np.unique(hsh)         # each row of J gets a unique label 
                                     
    idx_grp = [] # list of arrays that define the index groups
    for i in range(lbls.size):
        idx_grp.append(np.where(hsh==lbls[i])[0])

    obs_idx = [] # list of arrays giving the index groups observed at each
                 # given time interval
    for i in range(len(obs_pops)):
        obs_idx.append([])
        for j in np.unique(hsh[np.where(J[:,obs_pops[i]]==1)]):
            obs_idx[i].append(np.where(lbls==j)[0][0])            
    # note that we only store *where* the entry was found, i.e. its 
    # position in labels, not the actual label itself - hence we re-defined
    # the labels to range from 0 to len(idx_grp)

    return obs_idx, idx_grp


def get_obs_index_overlaps(idx_grp, sub_pops):
    """"  Computes index arrays for subpopulation overlaps. 

    Output index arrays of this function serve to quickly relate 
    subpopulations to their consituating index (or 'fate') groups.  

    Parameters
    ----------
    idx_grp: list of arrays
        observation scheme with keys 'sub_pops', 'obs_time', 'obs_pops'
    sub_pops: list or tuple of arrays
        index arrays for subpopulations (one array per subpopulation)

    Output
    ----------
    overlaps : list of arrays
        list of overlaps between subpopulations
    overlap_grp : list of arrays 
        list of index groups found in more than one subpopulation
    idx_overlap : list of arrays 
        list of subpops in which the corresponding index group is found
    """
    num_sub_pops = len(sub_pops) 
    num_idx_grps = len(idx_grp)

    idx_overlap = []
    idx = np.zeros(num_idx_grps, dtype=int)
    for j in range(num_idx_grps):
        idx_overlap.append([])
        for i in range(num_sub_pops):
            if np.any(np.intersect1d(sub_pops[i], idx_grp[j])):
                idx[j] += 1
                idx_overlap[j].append(i)
        idx_overlap[j] = np.array(idx_overlap[j])

    overlaps = [idx_grp[i] for i in np.where(idx>1)[0]]
    overlap_grp = [i for i in np.where(idx>1)[0]]
    idx_overlap = [idx_overlap[i] for i in np.where(idx>1)[0]]

    return overlaps, overlap_grp, idx_overlap


def comp_subpop_index_mats(sub_pops,idx_grp,overlap_grp,idx_overlap):
    """ Computes masks for observed, overlapping and cross-overlapping matrix parts"

    Parameters
    ----------
    sub_pops : list or tuple of arrays
        index arrays for subpopulations (one array per subpopulation)
    idx_grp : list of arrays
        arrays of index group memberships, one per index group  
    overlap_grp : list of arrays 
        list of index groups found in more than one subpopulation
    idx_overlap : list of arrays 
        list of subpops in which the corresponding index group is found

    Output
    ----------
    Om : p-by-p array or None
        mask for co-observed variables (instantaneous co-observation)
    Ovw : p-by-p array or None
        mask for overlaps across subpopulations 
    Ovc : p-by-p array or None
        mask for cross-overlaps (overlap across subpopualtions only) 
    """
    p = np.max([np.max(sub_pops[i]) for i in range(len(sub_pops))]) + 1

    if p < 10000:
        Om = np.zeros((p,p), dtype=bool)
        for i in range(len(sub_pops)):
            Om[np.ix_(sub_pops[i],sub_pops[i])] = True        
        Ovw = np.zeros((p,p), dtype=int)
        for i in range(len(sub_pops)):
            Ovw[np.ix_(sub_pops[i],sub_pops[i])] += 1
        Ovw = np.minimum(np.maximum(Ovw-1, 0),1)
        Ovw = np.asarray(Ovw, dtype=bool)

        Ovc = np.zeros((p,p), dtype=bool)
        for i in range(len(overlap_grp)):
            for j in range(len(idx_overlap[i])):
                Ovc[np.ix_(idx_grp[overlap_grp[i]], \
                    sub_pops[idx_overlap[i][j]])] = True
                Ovc[np.ix_(sub_pops[idx_overlap[i][j]], \
                    idx_grp[overlap_grp[i]])] = True
                Ovc[np.ix_(idx_grp[overlap_grp[i]], \
                    idx_grp[overlap_grp[i]])] = False

    else:
        Om, Ovw, Ovc = None, None, None
    
    return Om, Ovw, Ovc


###########################################################################
# Visualization
###########################################################################

def plot_slim(Qs, Om, lag_range, pars, idx_a, idx_b, traces, mmap, data_path):

    kl = len(lag_range)
    p,n = pars['C'].shape
    pa, pb = idx_a.size, idx_b.size
    idx_ab = np.intersect1d(idx_a, idx_b)
    idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
    idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]
    plt.figure(figsize=(20,10*np.ceil( (kl)/2)))
    for m in range(kl):
        m_ = lag_range[m] 
        Qrec = pars['C'][idx_a,:].dot(pars['X'][m*n:(m+1)*n, :]).dot(pars['C'][idx_b,:].T) 
        if m_ == 0:
            Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(pars['R'][idx_ab])
        plt.subplot(np.ceil( (kl)/2 ), 2, m+1, adjustable='box-forced')
        if mmap:
            Q = np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, mode='r', shape=(pa,pb))
        else:
            Q = Qs[m]
        plt.plot(Q[Om[m]].reshape(-1), Qrec[Om[m]].reshape(-1), '.')
        plt.title( ('m = ' + str(m_) + ', corr = ' + 
        str(np.corrcoef( Qrec[Om[m]].reshape(-1), (Qs[m][Om[m]]).reshape(-1) )[0,1])))
        if mmap:
            del Q
        plt.xlabel('true covs')
        plt.ylabel('est. covs')
    plt.show()
    plt.figure(figsize=(20,10))
    plt.plot(traces[0])
    plt.xlabel('iteration count')
    plt.ylabel('target loss')
    plt.title('loss function vs. iterations')
    plt.show()


def print_slim(Qs, Om, lag_range, pars, idx_a, idx_b, traces, mmap, data_path):

    kl = len(lag_range)
    p,n = pars['C'].shape
    pa, pb = idx_a.size, idx_b.size
    idx_ab = np.intersect1d(idx_a, idx_b)
    idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
    idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]
    for m in range(kl): 
        m_ = lag_range[m] 
        Qrec = pars['C'][idx_a,:].dot(pars['X'][m*n:(m+1)*n, :]).dot(pars['C'][idx_b,:].T) 
        if m_ == 0:
            Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(pars['R'][idx_ab])
        if mmap:
            Q = np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, mode='r', shape=(pa,pb))
        else:
            Q = Qs[m]
        print('m = ' + str(m_) + ', corr = ' + 
        str(np.corrcoef( Qrec[Om[m]].reshape(-1), (Qs[m][Om[m]]).reshape(-1) )[0,1]))
        if mmap:
            del Q

## printing process
# adapted from Matthew Johnson, 
# https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/text.py

round = (lambda x: lambda y: int(x(y)))(round)

# NOTE: datetime.timedelta.__str__ doesn't allow formatting the number of digits
def sec2str(seconds):
    hours, rem = divmod(seconds,3600)
    minutes, seconds = divmod(rem,60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours,minutes,round(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes,round(seconds))
    else:
        return '%0.2f' % seconds

def progprint_xrange(*args,**kwargs):
    xr = range(*args)
    return progprint(xr,total=len(xr),**kwargs)

def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    eta = sec2str(avgtime*(total-(idx+1)))
                    sys.stdout.write((
                        '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n'
                                % (numdigits,numdigits)) % (idx+1,total,avgtime,eta))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times),total))

