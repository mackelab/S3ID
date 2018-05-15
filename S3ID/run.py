import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from S3ID import main, print_slim, ObservationScheme, principal_angle

def run_default(alphas, b1s, a_decays, batch_sizes, max_zip_sizes, max_iters,
				pars_est, pars_true, n, 
				sso, obs_scheme, lag_range, idx_a, idx_b,
				y, Qs, Om, W, parametrizations,
				traces=[[], [], []], ts = [], verbose=True, dtype=np.float):
	""" Wrapper function for fitting LDS to data with the S3ID algorithm 

	Fits LDS in multiple stages, commonly seperated by different batch sizes.
	Recommended for large systems with long recordings, for e.g. running initial epochs
	with small batchsize before obtaining higher quality from large minibatches.  

	Parameters
	----------
	alphas : list or array
		ADAM parameters per fitting stage
	b1s :  list or array
		ADAM parameters per fitting stage
	a_decays : list or array
		ADAM parameters per fitting stage
	batch_sizes : list or array
		batch sizes per fitting stage
	max_zip_sizes : list or array
		epoch length per fitting stage
	max_iters : list or array
		maximum iterations length per fitting stage
	pars_est :  str or dict
		'default' for standard parameter estimat initialiation
	pars_true : dict
		ground-truth parameters (only for comparison of estimated subspace)
	n : int
		latent dimensionality 
	sso : boolean
		if True, assumes block-wise observation scheme (faster computations)
	obs_scheme : ObservationScheme object
		block-wise partial observation scheme
	lag_range : array
		array of time-lags to consider
	idx_a, idx_b : arrays
	y : array
		T-by-p array of observed data (unobserved entries will not be queried)
	Qs : list of arrays
		(masked) pair-wise time-lagged covariances
	Om : list of arrays
		mask for pair-wise time-lagged covariances
	W :  list or arrays
		list of inverse co-observation counts, as computed by ObservationScheme
	parametrizations : str
		str giving parametrization of latent dynamics model (linear, agnostic)
	traces : len-3 list of lists 
		optional; output traces of previous fitting runs
	ts : list
		optional; fitting durations of previous fitting runs	
	verbose : boolean 
		verbosity flag
	dtype : numpy datatype 
		datatype for numpy arrays

	Output
	----------
	pars_est : dict
		parameter estimate
	traces : list 
		list of fit diagnostics (loss & correlation of covariances over time)
	ts : list
		list of computation times
	"""
	T,p = y.shape

	return_aux = False # currently unused; 
	aux_init = None    # allowing to provide / keep ADAM moment and scaling auxiliary parameters

	if pars_est =='default':


		A = np.diag(np.linspace(0.89, 0.91, n),dtype=dtype) if parametrizations[0]=='ln' else None
		B = np.eye(n,dtype=dtype) if parametrizations[0]=='ln' else None
		Pi = np.eye(n,dtype=dtype) if parametrizations[0]=='ln' else None

		pars_est = {'A'  : A,
					'Pi' : B,
					'B'  : Pi,
					'C'  : np.asarray(np.random.normal(size=(p,n)), dtype=np.float32),
					'R'  : np.zeros(p,dtype=dtype),
					'X'  : np.zeros((len(lag_range)*n, n),dtype=dtype)} #pars_ssid['C'].dot(np.linalg.inv(M))} 


	assert len(alphas) == len(b1s)
	assert len(alphas) == len(a_decays) 
	assert len(alphas) == len(batch_sizes) 
	assert len(alphas) == len(max_zip_sizes) 
	assert len(alphas) == len(max_iters)
	assert len(alphas) == len(parametrizations)

	assert len(Qs) >= len(lag_range)
	assert len(Om) >= len(lag_range)
	assert len(W)  >= len(lag_range)

	assert len(traces) > 2

	for i in range(len(alphas)):

		parametrization = parametrizations[i]

		print('parametrization:', parametrization)
		if parametrization == 'ln':
			if pars_est['B'] is None:
				pars_est['Pi'] = (pars_est['Pi'] + pars_est['Pi'].T) / 2
				l = np.min( (np.real(np.linalg.eigvals(pars_est['Pi'])).min(), 0) )
				pars_est['B'] = np.linalg.cholesky(pars_est['Pi'] + (1e-10 - l) * np.eye(n))
			if pars_est['A'] is None:
				pars_est['A'] = np.linalg.lstsq(pars_est['X'][:(len(lag_range)-1)*n,:], pars_est['X'][n:len(lag_range)*n,:])[0]

		batch_size, max_zip_size, max_iter = batch_sizes[i], max_zip_sizes[i], max_iters[i]
		a, b1, b2, e = alphas[i], b1s[i], 0.99, 1e-8
		a_decay = a_decays[i]


		proj_errors = np.zeros((max_iter,np.min((n, pars_true['C'].shape[1]))+1))
		def pars_track(pars,t): 
		    C = pars[0]
		    proj_errors[t] = np.hstack((0, principal_angle(pars_true['C'], C)))
		pars_est, _, traces_, _, _, _, t = main(
			lag_range=lag_range,n=n,y=y, idx_a=idx_a,
			idx_b=idx_b,
			obs_scheme=obs_scheme,pars_init=pars_est,
			parametrization=parametrization, sso=sso,
			Qs=Qs, Om=Om, W=W,
			alpha=a,b1=b1,b2=b2,e=e,a_decay=a_decay,max_iter=max_iter,
			batch_size=batch_size,verbose=verbose, max_epoch_size=max_zip_size,
			pars_track=pars_track,dtype=dtype,
			return_aux=return_aux,aux_init=aux_init)
		traces[0].append(traces_[0])
		traces[1].append(traces_[1])
		traces[2].append(proj_errors.copy())
		ts.append(t)

		print_slim(Qs,Om,lag_range,pars_est,idx_a,idx_b,traces_,False,None)
		print('fitting time was ', t, 's')
		plt.plot(proj_errors[:,1:])
		plt.show()

	return pars_est, traces, ts
