'''
get_zdFF.py calculates standardized dF/F signal based on calcium-idependent 
and calcium-dependent signals commonly recorded using fiber photometry calcium imaging

Ocober 2019 Ekaterina Martianova ekaterina.martianova.1@ulaval.ca

Reference:
  (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry 
      to Record Neural Activity in Freely Moving Animal. J. Vis. Exp. 
      (152), e60278, doi:10.3791/60278 (2019)
      https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving

'''
import numpy as np
from sklearn.linear_model import Lasso
import itertools
import pandas as pd
import scipy

from sklearn.model_selection import train_test_split
from utils_models import ksplit_X_y, simple_metric, fp_corrected_metric
from functools import partial

from utils_signal import robust_filter


def jove_fit_reference(reference, signal, smooth_win=10, remove=200,
                       use_raw=True, lambd=5e4, porder=1, itermax=50, robust_base=False):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-independent
    and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer, switch to 1 to use raw signal
        remove: the beginning of the traces with a big slope one would like to remove, integer
        use_raw: whether to use raw data rather than smoothened when fitting reference
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        z_reference: z-score reference channel
        z_signal: z-score signal data
        z_reference_fitted: fitted z-score reference
    '''
    # Smooth Signal
    raw_reference, raw_signal = reference, signal
    smoothened_reference = smooth_signal(reference, smooth_win)
    smoothened_signal = smooth_signal(signal, smooth_win)
    # Find the baseline
    if robust_base:
        r_base = scipy.ndimage.percentile_filter(smoothened_reference, 50, size=400)
        s_base = robust_filter(smoothened_signal, window=200, buffer=False, method=12)[:, 0]
    else:
        r_base=airPLS(smoothened_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
        s_base=airPLS(smoothened_signal,lambda_=lambd,porder=porder,itermax=itermax)
    # Remove the baseline and the beginning of the recordings
    if use_raw:
        reference = (raw_reference[remove:] - r_base[remove:])
        signal = (raw_signal[remove:] - s_base[remove:])
    else:
        reference = (smoothened_reference[remove:] - r_base[remove:])
        signal = (smoothened_signal[remove:] - s_base[remove:])
    # Standardize signals
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)

    ### Fit reference signal to calcium signal using linear regression
    from sklearn.linear_model import Lasso
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(z_reference)
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
    return z_reference, z_signal, z_reference_fitted


# Apply the smoothing baseline removal together and then split the data in k_fold
# Does it make more sense to split first (more edge effect) or preprocess first?
# make method Jove_preprocess(**kwargs)
def jove_preprocess(reference, signal, smooth_win=10, remove=200,
                       use_raw=True, lambd=5e4, porder=1, itermax=50):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent
    and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer, switch to 1 to use raw signal
        remove: the beginning of the traces with a big slope one would like to remove, integer
        use_raw: whether to use raw data rather than smoothened when fitting reference
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        z_reference: z-score reference channel
        z_signal: z-score signal data
        z_reference_fitted: fitted z-score reference
    '''
    # Smooth Signal
    raw_reference, raw_signal = reference, signal
    smoothened_reference = smooth_signal(reference, smooth_win)
    smoothened_signal = smooth_signal(signal, smooth_win)
    # Find the baseline
    r_base=airPLS(smoothened_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(smoothened_signal,lambda_=lambd,porder=porder,itermax=itermax)
    # Remove the baseline and the beginning of the recordings
    if use_raw:
        reference = (raw_reference[remove:] - r_base[remove:])
        signal = (raw_signal[remove:] - s_base[remove:])
    else:
        reference = (smoothened_reference[remove:] - r_base[remove:])
        signal = (smoothened_signal[remove:] - s_base[remove:])
    # Standardize signals
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)
    return z_reference, z_signal


def jove_find_best_param(reference, signal, smooth_win=10, remove=200,
                  use_raw=True, itermax=50, k_split=5):
    ## TODO: deprecated could lead to non-smooth filtering
    # Integrate vflow for future
    param_grid = {'lambd': [5e1, 5e2, 5e3, 5e4, 5e5], 'porder': [1, 2]}
    params = list(param_grid.keys())
    param_product = itertools.product(*[param_grid[pr] for pr in param_grid])
    result_df = []
    for pgrid in param_product:
        grid_vals = {params[i]: pgrid[i] for i in range(len(params))}
        z_reference, z_signal = jove_preprocess(reference, signal, smooth_win,
                                                remove, use_raw, itermax=itermax, **grid_vals)
        X = z_reference.reshape((-1, 1))
        y = z_signal
        metrics = {'resid_std': simple_metric, 'r2': partial(fp_corrected_metric, method='r2'),
                   'var_ratio': partial(fp_corrected_metric, method='explained_variance')}
        sort_metric = 'resid_std'
        results = []
        for k_X, k_y in ksplit_X_y(X, y, k_split):
            k_X_train, k_X_test, k_y_train, k_y_test = train_test_split(k_X, k_y, test_size=0.3, shuffle=False)
            lin = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                        positive=True, random_state=9999, selection='random')
            lin.fit(k_X_train, k_y_train)
            k_y_pred = lin.predict(k_X_test).ravel()
            results.append([metrics[m](k_y_test, k_y_pred) for m in metrics])
        iresult_df = pd.DataFrame(results, columns=list(metrics.keys()))
        iresult_df['fold'] = np.arange(k_split)
        iresult_df[params] = pgrid
        result_df.append(iresult_df)
    result_df = pd.concat(result_df, axis=0)
    final = result_df.drop(columns=['fold']).groupby(params, as_index=False).mean().sort_values(sort_metric, ascending=False)
    #result_df.drop(columns=['fold']).groupby(params).agg(['mean', 'std'])
    pgrid_star = final.loc[0, params].to_dict()
    return result_df, pgrid_star


def get_zdFF(reference, signal, smooth_win=10, remove=200, use_raw=True, lambd=5e4, porder=1, itermax=50):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent
    and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer, switch to 1 to use raw signal
        remove: the beginning of the traces with a big slope one would like to remove, integer
        use_raw: whether to use raw data rather than smoothened when fitting reference
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        z_reference: z-score reference channel
        z_signal: z-score signal data
        z_reference_fitted: fitted z-score reference
    '''
    # Smooth Signal
    raw_reference, raw_signal = reference, signal
    smoothened_reference = smooth_signal(reference, smooth_win)
    smoothened_signal = smooth_signal(signal, smooth_win)
    # Find the baseline
    r_base=airPLS(smoothened_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(smoothened_signal,lambda_=lambd,porder=porder,itermax=itermax)
    # Remove the baseline and the beginning of the recordings
    if use_raw:
        reference = (raw_reference[remove:] - r_base[remove:])
        signal = (raw_signal[remove:] - s_base[remove:])
    else:
        reference = (smoothened_reference[remove:] - r_base[remove:])
        signal = (smoothened_signal[remove:] - s_base[remove:])
    # Standardize signals
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)

    ### Fit reference signal to calcium signal using linear regression
    from sklearn.linear_model import Lasso
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(z_reference)
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
    return z_signal - z_reference_fitted


def get_zdFF_old(reference,signal,smooth_win=10,remove=200,lambd=5e4,porder=1,itermax=50, raw=False):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent
    and calcium-dependent signals

    Input
      reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
      signal: calcium-dependent signal (usually 465-490 nm excitation for
                   green fluorescent proteins, or ~560 nm for red), 1D array
      smooth_win: window for moving average smooth, integer
      remove: the beginning of the traces with a big slope one would like to remove, integer
      Inputs for airPLS:
      lambd: parameter that can be adjusted by user. The larger lambda is,
              the smoother the resulting background, z
      porder: adaptive iteratively reweighted penalized least squares for baseline fitting
      itermax: maximum iteration times
    Output
      zdFF - z-score dF/F, 1D numpy array
    Albert: modifed by adding raw version without using dff
    '''


    # Smooth signal
    reference = smooth_signal(reference, smooth_win)
    signal = smooth_signal(signal, smooth_win)

    # Remove slope using airPLS algorithm
    r_base=airPLS(reference, lambda_=lambd, porder=porder, itermax=itermax)
    s_base=airPLS(signal, lambda_=lambd, porder=porder, itermax=itermax)

    # Remove baseline and the begining of recording
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])

    reference_orig, signal_orig = reference, signal
    # Standardize signals
    reference = (reference - np.median(reference)) / np.std(reference)
    sig_pow, sig_med = np.std(signal), np.median(signal)
    signal = (signal - np.median(signal)) / np.std(signal)

    # Align reference signal to calcium signal using non-negative robust linear regression
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
              positive=True, random_state=9999, selection='random')
    n = len(reference)
    if raw:
        lin.fit(reference_orig.reshape(n, 1), signal_orig.reshape(n, 1))
        reference_orig = lin.predict(reference_orig.reshape(n, 1)).reshape(n,)
        return signal - reference_orig
    else:
        lin.fit(reference.reshape(n, 1), signal.reshape(n, 1))
        reference = lin.predict(reference.reshape(n,1)).reshape(n,)

        # z dFF
        zdFF = (signal - reference)
        # this will result in lower signal when there is large motion artifacts etc.
        return zdFF


def get_f0_Martianova_jove(reference,signal,smooth_win=40,remove=0,lambd=5e4,porder=1,itermax=50):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent
    and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer
        remove: the beginning of the traces with a big slope one would like to remove, integer
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        zdFF - z-score dF/F, 1D numpy array
    '''

    # Smooth signal
    reference = smooth_signal(reference, smooth_win)
    signal = smooth_signal(signal, smooth_win)

    # Remove slope using airPLS algorithm
    r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax)

    # Remove baseline and the begining of recording
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])

    # Standardize signals
    # reference = (reference - np.median(reference)) / np.std(reference)
    # signal = (signal - np.median(signal)) / np.std(signal)

    # Align reference signal to calcium signal using non-negative robust linear regression
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(reference)
    lin.fit(reference.reshape(n,1), signal.reshape(n,1))
    reference = lin.predict(reference.reshape(n,1)).reshape(n,)

    # z dFF
    return reference

def get_dFF(reference,signal,smooth_win=10,remove=200,lambd=5e4,porder=1,itermax=50):
    '''
    Calculates z-score dF/F signal based on fiber photometry calcium-idependent
    and calcium-dependent signals

    Input
        reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
        signal: calcium-dependent signal (usually 465-490 nm excitation for
                     green fluorescent proteins, or ~560 nm for red), 1D array
        smooth_win: window for moving average smooth, integer
        remove: the beginning of the traces with a big slope one would like to remove, integer
        Inputs for airPLS:
        lambd: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: maximum iteration times
    Output
        zdFF - z-score dF/F, 1D numpy array
    '''


    # Smooth signal
    reference = smooth_signal(reference, smooth_win)
    signal = smooth_signal(signal, smooth_win)

    # Remove slope using airPLS algorithm
    r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax)

    # Remove baseline and the begining of recording
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])

    # Standardize signals
    # reference = (reference - np.median(reference)) / np.std(reference)
    # signal = (signal - np.median(signal)) / np.std(signal)

    # Align reference signal to calcium signal using non-negative robust linear regression
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(reference)
    lin.fit(reference.reshape(n,1), signal.reshape(n,1))
    reference = lin.predict(reference.reshape(n,1)).reshape(n,)

    # z dFF
    zdFF = (signal - reference)

    return zdFF


def smooth_signal(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """
    
    import numpy as np

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]


'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares

This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively 
reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

Description from the original documentation:
Baseline drift always blurs or even swamps signals and deteriorates analytical 
results, particularly in multivariate analysis.  It is necessary to correct baseline 
drift to perform further data analysis. Simple or modified polynomial fitting has 
been found to be effective in some extent. However, this method requires user 
intervention and prone to variability especially in low signal-to-noise ratio 
environments. The proposed adaptive iteratively reweighted Penalized Least Squares
(airPLS) algorithm doesn't require any user intervention and prior information, 
such as detected peaks. It iteratively changes weights of sum squares errors (SSE) 
between the fitted baseline and original signals, and the weights of SSE are obtained 
adaptively using between previously fitted baseline and original signals. This 
baseline estimator is general, fast and flexible in fitting baseline.


LICENCE
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is, 
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z
