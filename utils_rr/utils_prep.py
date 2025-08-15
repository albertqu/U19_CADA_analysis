""" 
This file contains function that handles preprocessing of Restaurant Row task data.
"""

from utils_rr.utils_dataframe import df_lagmat
import numpy as np
import pandas as pd 


# TODO: think about how to calculate number of reward collected so far

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

def transform_rr_behavioral_features(bdf):
    """ 
    This function transforms a range of behavioral features and prepare engineered 
    features for modeling Restaurant Row (RR) task data.

    offer: 
        offer_rank: Rank of the offer (1-4), consistent with order statistics (ASC)
        low_offer: offer with rank 1 and 2
        high_offer: offer with rank 3 and 4
    choice:
        hasOutcome: choice resulted in an outcome, implying a committed trial
        ACC / REJ
        commit: accept choice with no quit (clean accept), together this gives: 
            commit=1, accept=1 -> clean accept, commit=0, accept=1 -> quit, 
            commit=0, accept=0 -> reject
        low_offer_accept: accept with low offer
        high_offer_reject: reject with high offer
    time/latency features:
        log_hall_time: log of hall time
        wait_time: time waited after accept
        log_wait_time: log of wait time

    Note: importantly, for this function, we use bonsai_quit, as opposed to slp_quit for decision
    categorization; need to also consider that some time a quit is merely a reject

    Input:
        bdf: DataFrame
            A pandas DataFrame containing behavioral data for all sessions of the RR task.
            (assuming containing slp columns)
    Output:
        session_bdf: DataFrame
            DataFrame appended in place with new features
    """
    slp_mod = 'slp_'
    slp_str = lambda x: slp_mod + x
    # offer features
    session_bdf = bdf
    session_bdf['offer_rank'] = session_bdf['offer_prob'].rank(method='dense').astype(float)
    median_rank = np.median(session_bdf['offer_rank'].unique())
    session_bdf['low_offer'] = (session_bdf['offer_rank'] <= median_rank).astype(float)
    session_bdf['high_offer'] = (session_bdf['offer_rank'] > median_rank).astype(float)
    # choice features
    session_bdf['commit'] = session_bdf['outcome'].notnull().astype(float)
    session_bdf['ACC'] = ((session_bdf['commit'] == 1) | (session_bdf[slp_str('accept')] == 1)).astype(float)
    session_bdf['REJ'] = ((session_bdf['commit'] == 0) & (session_bdf[slp_str('accept')] == 0)).astype(float)
    session_bdf['low_offer_accept'] = ((session_bdf['low_offer'] == 1) 
                                       & (session_bdf['ACC'] == 1)).astype(float)
    session_bdf['high_offer_reject'] = ((session_bdf['high_offer'] == 1) 
                                        & (session_bdf['REJ'] == 1)).astype(float)
    # time/latency features
    session_bdf['log_hall_time'] = np.log(session_bdf[slp_str('hall_time')])
    session_bdf['wait_time'] = np.nan
    waited_sel = session_bdf['commit'] == 1
    session_bdf.loc[waited_sel, 'wait_time'] =  session_bdf.loc[waited_sel, 'outcome'] - session_bdf.loc[waited_sel, slp_str('choice_time')]
    quit_sel = (session_bdf['commit'] == 0) & (session_bdf['ACC'] == 1)
    session_bdf.loc[quit_sel, 'wait_time'] = session_bdf.loc[quit_sel, 'quit'] - session_bdf.loc[quit_sel, slp_str('choice_time')]
    session_bdf['log_wait_time'] = np.log(session_bdf['wait_time'])
    return session_bdf

def add_time_wise_history_features(session_bdf, dt=120, inplace=True): 
    """ Method to add past outcome information as features to data, 
    assumes `transform_rr_behavioral_features` has been called.
        data: pd.DataFrame
        dt: float, time periods to include in seconds, by default 120
    """
    data = session_bdf
    slp_mod = 'slp_'
    slp_str = lambda x: slp_mod + x
    if not inplace:
        data = data.reset_index(drop=True)
    col_arg1 = f'past_rew_{dt}s'
    data[col_arg1] = 0.0
    col_arg2 = f'past_accept_{dt}s'
    data[col_arg2] = 0.0
    col_arg3 = f'past_waits_{dt}s'
    data[col_arg3] = 0.0
    for i in range(len(data)):
        t_i = data.loc[i, 'tone_onset']
        s_i = t_i - dt
        n_rew = np.sum((data['outcome'] >= s_i) & (
            data['outcome'] < t_i) & (data['reward'] == 1))
        choice_sel = (data[slp_str('choice_time')] >= s_i) & (data[slp_str('choice_time')] < t_i)
        n_accept = np.sum(choice_sel & (data['ACC'] == 1))
        total_wait = np.sum(data.loc[choice_sel & (data['ACC'] == 1), 'wait_time'])
        data.loc[i, col_arg1] = n_rew
        data.loc[i, col_arg2] = n_accept
        data.loc[i, col_arg3] = total_wait
    return data

def regularize_restaurant_trial_structure(session_bdf):
    """
    Standardizes restaurant trial data to ensure consistent sequence structure.
    
    This function takes behavioral data from an RR (Restaurant Row) experiment and 
    reorganizes/relabels it to ensure trials follow a consistent pattern where:
    1. Restaurants strictly follow the sequence [1, 2, 3, 4, 1, 2, 3, 4, ...] 
    2. Each complete cycle through all 4 restaurants is labeled as one "lap"
    3. Trial numbers are sequential and start from 1

    Note: This standardization is useful for analyses that require consistent trial 
    structure and accurate lap counting, particularly when the original data
    might contain skipped trials or irregular restaurant sequences.
    
    Parameters
    ----------
    session_bdf : pandas.DataFrame
        Input dataframe containing RR behavior data. Expected columns include 
        'trial', 'restaurant', 'tone_onset', 'lapIndex'
    
    Returns
    -------
    restaurant_df : pandas.DataFrame
        A modified dataframe with updated trial structure:
          - 'trial': Sequential trial numbers starting from 1.
          - 'restaurant': Reassigned restaurant labels following the [1, 2, 3, 4] cycle.
          - 'lapIndex': Updated lap index computed as (trial - 1) // 4.
    """
    restaurant_df = session_bdf
    trial_inds = calculate_restaurant_sequence_index(restaurant_df['restaurant'].values)
    restaurant_df['trial_ind'] = trial_inds
    restaurant_df.set_index('trial_ind', inplace=True)
    restaurant_df = restaurant_df.reindex(np.arange(max(trial_inds) + 1))
    restaurant_df['seqTrial'] = restaurant_df.index + 1
    restaurant_df.reset_index(drop=True, inplace=True)
    restaurant_df['restaurant'] = (restaurant_df['trial'] - 1) % 4 + 1
    restaurant_df['lapIndex'] = (restaurant_df['trial'] - 1) // 4
    animal, session = restaurant_df['animal'].iloc[0], restaurant_df['session'].iloc[0]
    restaurant_df['animal'] = restaurant_df['animal'].fillna(animal)
    restaurant_df['session'] = restaurant_df['session'].fillna(session)
    return restaurant_df

def get_rr_lagged_trial_features(session_bdf, lag=12):
    """ 
    This function extracts lagged trial features from RR task data.
    Specifically, the function back lags ['offer_rank', 'ACC', 
    'commit', 'reward', 'low_offer_accept', 'high_offer_reject'] by `lag` trials.
    """
    lagdf = df_lagmat(session_bdf, ['offer_rank', 'ACC', 'commit', 'reward', 
                                    'low_offer_accept', 'high_offer_reject'], back=lag)
    return pd.concat([session_bdf, lagdf], axis=1)

def calculate_trial_history_feature(session_bdf, lag=12):
    """
    This function first lags trial features, then calculates several 
    complex trial-history-based features.
    1) past rates (in t trials, of past tau seconds):
    reward rate, accept rate, commit rate
    low offer ACC rate, high offer REJ rate
    2) immediate history:
    prev_trial_offer, prev_lap_offer
    """
    features = ['offer_rank', 'ACC', 'commit', 'reward', 
                'low_offer_accept', 'high_offer_reject']
    lagdf = df_lagmat(session_bdf, features, back=lag)
    hist_data = {}
    for feat in features:
        lagcols = [f'{feat}__b{i}' for i in range(1, lag+1)]
        vals = lagdf[lagcols].values
        vals[np.all(np.isnan(vals), axis=1), :] = 0
        hist_data[f'past_{feat}_rate'] = np.nanmean(vals, axis=1)
    hist_data['prev_lap_offer'] = lagdf['offer_rank__b4'].values
    hist_data['prev_trial_offer'] = lagdf['offer_rank__b1'].values
    hist_data['prev_trial_reward'] = lagdf['reward__b1'].values
    hist_data['prev_lap_reward'] = lagdf['reward__b4'].values
    hist_data['reward_collected'] = lagdf['reward__b1'].fillna(0).cumsum()
    return pd.concat([session_bdf, lagdf, pd.DataFrame(hist_data)], axis=1)

def combine_rr_functions(session_bdf, funcs):
    """ This function takes in multiple processing functions and performs them in sequence.
    funcs: list of tuple: (function, kwargs)
    """
    for f, kwargs in funcs:
        session_bdf = f(session_bdf, **kwargs)
    return session_bdf