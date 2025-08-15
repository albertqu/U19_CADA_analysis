import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat

##############################################
######### RR Behavior Data testing  ##########
##############################################

def calculate_restaurant_sequence_index(restaurants):
    """Given a sequence of restaurants, infer trial number, assuming no skips.
    Example:
    >>> calculate_structured_trial_index([1, 3, 4, 1, 2, 3, 1])
    [0, 2, 3, 4, 5, 6, 8]
    """
    assert len(restaurants) > 0, "empty data"
    trial_ids = [-1] * len(restaurants)
    curr_i = 0
    r = restaurants[0]
    if r != 1:
        diff = r - 1
        curr_i += diff
    trial_ids[0] = curr_i
    prev_r = r
    for i in range(1, len(restaurants)):
        r = restaurants[i]
        if r == prev_r:
            diff = 4
        else:
            diff = (r - prev_r) % 4
        curr_i += diff
        trial_ids[i] = curr_i
        prev_r = r
    return trial_ids

def check_restaurant_structure(session_bdf):
    restaurant_df = session_bdf[['trial', 'restaurant', 'tone_onset', 'lapIndex']].reset_index(drop=True)
    restaurant_df['pred_restaurant_next'] = restaurant_df['restaurant'] % 4 + 1
    restaurant_df['actual_restaurant_next'] = restaurant_df['restaurant'].shift(-1)
    restaurant_df.dropna(subset=['actual_restaurant_next', 'pred_restaurant_next'], inplace=True)
    return np.all(restaurant_df['pred_restaurant_next'] == restaurant_df['actual_restaurant_next'].astype(int))

def test_calculate_restaurant_sequence_index():
    test_cases = [
        [1, 3, 4, 1, 2, 3, 1],
        [1, 2, 4, 2, 4, 4, 3, 1, 2, 3, 4, 1, 2, 3, 4, 2]
    ]
    total = 0
    for test_arg in test_cases:
        result = calculate_restaurant_sequence_index(test_arg)
        correct = np.allclose(np.array(result) % 4 + 1, test_arg)
        total += int(correct)
    print('Accuracy rate: ', total / len(test_cases))

def test_RR_decison_categorization(sleap_bdf):
    # This testing established that when classifying decision, use slp_accept + outcome
    # edge case: first use outcome: commit=1, (accept=1, clean accept), otherwise only when slp_accept=1 is quit trials
    # objective, understand how to best assign commit and accept behavior
    test_df = sleap_bdf[['animal', 'session', 'trial', 'accept', 'slp_accept', 'choice', 'slp_choice_time', 
                        'quit', 'slp_quit_time', 'outcome', 'tone_onset', 'lastTrial']].reset_index(drop=True)
    test_df['next_tone'] = test_df['tone_onset'].shift(-1)
    test_df.loc[test_df['lastTrial'], 'next_tone'] = np.nan
    # # when bonsai accept=0, sometimes there is still outcome? possibly mislabeled: slp_accept is CORRECT
    # test_df.loc[(test_df['accept'] == 0) & (test_df['outcome'].notnull())]['slp_accept']

    # when both bonsai and sleap output 1, bonsai quit is golden standard! (rare for slp quit=0 yet bonsai quit=1, and bonsai_quit is correct)
    # test_df[(test_df['accept'] == 1) & (test_df['slp_accept'] == 1) 
    #         & (test_df['quit'].isnull()) & (test_df['slp_quit_time'].notnull())]

    # also it is extremely rare when slp_accept=1, accept=0, and slp_quit=1
    # test_df[(test_df['accept'] == 0) & (test_df['slp_accept'] == 1) & (test_df['slp_quit_time'].isnull())]

    # do bonsai accept ever do better than slp_accept? (ie outcome is not null) yes but only 7 trials
    # this is when sleap based method mislabeled an accept trial as reject
    # test_df[(test_df['accept'] == 1) & (test_df['slp_accept'] == 0) & (test_df['outcome'].notnull())]

    # do bonsai accept=1, slp_accept=0, actually have lots of commmit trials? if quit is not null, then 
    # for sure there is no outcome
    # test_df[(test_df['accept'] == 1) & (test_df['slp_accept'] == 0) & (test_df['quit'].notnull())]

    test_df['slp_quitT'] = test_df['slp_quit_time'].notnull()
    test_df['quitT'] = test_df['quit'].notnull()
    test_df['hasOutcome'] = test_df['outcome'].notnull()
    test_df[['slp_accept', 'accept', 'slp_quitT', 'quitT', 'hasOutcome', 
            'trial']].groupby(['slp_accept', 'accept', 'hasOutcome']).agg({'trial':'count', 
                                                                            'slp_quitT': 'mean', 
                                                                            'quitT': 'mean'}).reset_index().dropna().sort_values(['slp_accept',
                                                                                                                                'hasOutcome'])

    test_df[['slp_accept', 'accept', 'slp_quitT', 'quitT', 'hasOutcome', 
            'trial']].groupby(['slp_accept', 'accept', 'slp_quitT', 
                                'quitT']).agg({'trial':'count', 
                                            'hasOutcome': 'mean'}).reset_index().dropna().sort_values(['slp_accept', 'quitT'])


###############################################
######### Data Frame function testing  ########
###############################################

def double_lagmat_opt(df, forw, back):
    lm_forw = lagmat(df.values, maxlag=forw, trim='backward', original="in")[:, :-len(df.columns)]
    lm_back = lagmat(df.values, maxlag=back, trim='forward', original="ex")
    cols = [f"{c}__f{-i}" for i in range(-forw, 0) for c in df.columns] + [
            f"{c}__b{i}" for i in range(1, back+1) for c in df.columns
        ]
    ncol = len(df.columns)
    for i in range(1, back+1):
        lm_back[:i, (i-1) * ncol: i * ncol] = np.nan
    for i in range(-forw, 0):
        lm_forw[i:, (forw+i) * ncol: (forw+i+1) * ncol] = np.nan
    res = pd.DataFrame(np.concatenate((lm_forw, lm_back), axis=1), columns=cols)
    return res

def single_lagmat_opt(df, forw, back):
    if forw == 0 and back == 0:
        return None
    data = lagmat(df, maxlag=forw+back, trim='None', original='in')
    data = data[forw:len(data)-back]
    ncol = len(df.columns)
    for i in range(1, back+1):
        data[:i, (i-1) * ncol + (forw+1) * ncol: i * ncol + (forw+1) * ncol] = np.nan
    for i in range(-forw, 0):
        data[i:, (forw+i) * ncol: (forw+i+1) * ncol] = np.nan
    lagcols = [f"{c}__f{-i}" for i in range(-forw, 0) for c in df.columns] + [
        f"{c}__b{i}" for i in range(1, back+1) for c in df.columns
    ]
    if forw and back:
        res = pd.DataFrame(np.concatenate([data[:, :forw*ncol], data[:, -back*ncol:]], axis=1),
                        columns=lagcols)
    elif forw:
        res = pd.DataFrame(data[:, :forw*ncol], columns=lagcols)
    else:
        res = pd.DataFrame(data[:, -back*ncol:], columns=lagcols)
    return res

def test_lag_feature_operations():
    """ 
    This test compare functions that perform lag for multiple columns
    1. either do one lagmat run
    2. pd shift (BAD)
    lagmat outperforms pd.shift by 4x in this example
    """
    # option 1: lagmat
    def lagmat_opt(df, L):
        lm = lagmat(df.values, maxlag=L, trim='forward', original="ex")
        for i in range(1, L+1):
            lm[:i, i-1] = np.nan  
        res = pd.DataFrame(lm, columns = [f'b{i}' for i in range(1, L+1)])
        return res

    # option 2: for with pd shift
    def pdshift_opt(df, L):
        vs = [None] * L
        for i in range(1, L+1):
            vf = df.shift(i)
            vf.name = f'b{i}'
            vs[i-1] = vf
        return pd.concat(vs, axis=1)

    import time
    d = 1
    n = 5000
    N = 1000
    L = 10
    df = pd.DataFrame(np.random.random((n, d)), columns=[f'c{i}' for i in range(d)])
    start = time.time()
    for _ in range(N):
        lagmat_opt(df, L)
    end = time.time()
    print(end - start)

    start = time.time()
    for _ in range(N):
        pdshift_opt(df, L)
    end = time.time()
    print(end - start)

    """
    More broadly, we also compared runtime between using single lagmat and 2 lagmat separately for front
    and back lags. The result is also indistinguishable.
    """

    start = time.time()
    for _ in range(N):
        #lagmat_opt(df, L)
        single_lagmat_opt(df, L, L)
    end = time.time()
    print(end - start)

    start = time.time()
    for _ in range(N):
        double_lagmat_opt(df, L, L)
    end = time.time()
    print(end - start)