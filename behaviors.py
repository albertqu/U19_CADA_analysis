# System

# Data
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
# Plotting
import matplotlib.pyplot as plt

# Utils
from utils import *


#######################################################
###################### Analysis #######################
#######################################################
# ALL TRIALS ARE 1-indexed!!
def get_action_outcome_latencies(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    outcome_times = get_behavior_times(mat, 'outcome')
    ipsi, contra = ("right", "left") if np.array(access_mat_with_path(mat, "notes/hemisphere")).item() \
        else ("left", "right")
    ipsi_choice_trials = access_mat_with_path(mat, f'trials/{ipsi}_in_choice', ravel=True, dtype=np.int)
    ipsi_choice_time = access_mat_with_path(mat, f'time/{ipsi}_in_choice', ravel=True)
    contra_choice_trials = access_mat_with_path(mat, f'trials/{contra}_in_choice', ravel=True, dtype=np.int)
    contra_choice_time = access_mat_with_path(mat, f'time/{contra}_in_choice', ravel=True)
    ipsi_lat = outcome_times[ipsi_choice_trials-1] - ipsi_choice_time
    contra_lat = outcome_times[contra_choice_trials - 1] - contra_choice_time
    return ipsi_lat, contra_lat


def get_center_port_stay_time(mat):
    # assuming len(center_in_time) == total trial
    center_in_time = access_mat_with_path(mat, 'glml/time/center_in', ravel=True)
    center_out_time = access_mat_with_path(mat, 'glml/time/execute', ravel=True)
    center_out_trial = access_mat_with_path(mat, 'glml/trials/execute', ravel=True, dtype=np.int)
    key_center_in_time = center_in_time[center_out_trial-1]
    return center_out_time - key_center_in_time


# TODO: take into account of possibility of duplicates
def get_trial_outcome_laterality(mat, as_array=False):
    """
    Returns 0-indexed trials with different lateralities
    :param mat:
    :return: ipsi, contra trials respectively
    """

    lateralities = np.zeros(get_trial_num(mat))
    lat_codes = {'ipsi': 1, "contra": 2, "None": 0}
    for side in 'ipsi', 'contra':
        rew = access_mat_with_path(mat, f'glml/trials/{side}_rew', ravel=True, dtype=np.int) - 1
        unrew = access_mat_with_path(mat, f'glml/trials/{side}_unrew', ravel=True, dtype=np.int) - 1
        lateralities[np.concatenate((rew, unrew))] = lat_codes[side]
    if as_array:
        return lateralities
    return decode_trial_behavior(lateralities, lat_codes)


def get_trial_outcomes(mat, as_array=False):
    """ TODO: remember 1-indexed
    Returns 0-indexed trials with different outcomes
    1.2=reward, 1.1 = correct omission, 2 = incorrect, 3 = no choice,  0: undefined
    :param mat:
    :param as_array: if True, returns array instead of dict
    :return: rewarded, unrewarded trials
    Not using boolean due to possibility of an no-outcome trial
    unrewarded: (x-1.2)^2 * (x-3) < 0, rewarded: x == 1.2
    """
    outcomes = access_mat_with_path(mat, f'glml/value/result', ravel=True)
    if as_array:
        return outcomes
    outcomes = decode_trial_behavior(outcomes, {'No choice': 3, 'Incorrect': 2, 'Correct Omission': 1.1,
                                            'Rewarded': 1.2})
    outcomes['Unrewarded'] = outcomes['Incorrect'] | outcomes['Correct Omission']
    return outcomes


def get_trial_features(mat, feature, as_array=False):
    """
    :param mat:
    # {} syntax for selections of features
    :param feature: time lag coded as {t-k,...} (no space in between allowed)
    :return:
    """
    fpast = trial_vector_time_lag

    results = {}
    N_trial = get_trial_num(mat)
    if feature == 'R{t-2,t-1}':
        trial_outcomes = get_trial_outcomes(mat)
        outcomes = ['Unrewarded', 'Rewarded']
        for oi in outcomes:
            for oj in outcomes:
                results[oi[0] + oj[0]] = np.logical_and.reduce([fpast(trial_outcomes[oi], -2),
                                                                fpast(trial_outcomes[oj], -1)])
    elif feature == 'O{t-2,t-1}':
        trial_outcomes = get_trial_outcomes(mat)
        outcomes = ['Incorrect', 'Correct Omission', 'Rewarded']
        for oi in outcomes:
            for oj in outcomes:
                results[oi[0] + oj[0]] = np.logical_and.reduce([fpast(trial_outcomes[oi], -2),
                                                                fpast(trial_outcomes[oj], -1)])
    elif feature == 'A{t-1,t}':
        trial_laterality = get_trial_outcome_laterality(mat)
        lateralities = ('ipsi', 'contra')
        for il in lateralities:
            for jl in lateralities:
                stay = 'stay' if (il == jl) else 'switch'
                results[jl + '_' + stay] = np.logical_and.reduce([fpast(trial_laterality[il], 0),
                                                                  fpast(trial_laterality[jl], -1)])
    elif feature == 'ITI':
        itis = access_mat_with_path(mat, "glml/trials/ITI", ravel=True)
        intervals = [(1.05, 4), (0.65, 1.05), (0.5, 0.65), (0, 0.5)]
        results = {itvl: (itis > itvl[0]) & (itis <= itvl[1]) for itvl in intervals}

    elif feature == 'O':
        trial_outcomes = get_trial_outcomes(mat)
        outcomes = ['Incorrect', 'Correct Omission', 'Rewarded']
        results = {oo: trial_outcomes[oo] for oo in outcomes}

    elif feature == 'A':
        trial_laterality = get_trial_outcome_laterality(mat)
        lateralities = ('ipsi', 'contra')
        results = {lat: trial_laterality[lat] for lat in lateralities}

    elif feature == 'ITI_raw':
        assert as_array, 'raw value yields no boolean'
        return access_mat_with_path(mat, "glml/trials/ITI", ravel=True)
    else:
        raise NotImplementedError(f"Unimplemented {feature}")

    if as_array:
        feat_array = np.full(N_trial, 'NA')
        for rf in results:
            feat_array[results[rf]] = rf
        return feat_array
    return results


def get_trial_num(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    return np.prod(access_mat_with_path(mat, 'trials/ITI').shape)


def decode_trial_behavior(arr, code):
    return {c: arr == code[c] for c in code}


def event_parse_lags(event):
    event = event.replace(" ", "")
    evt_split = event.split("{")
    if len(evt_split) > 1:
        lagstr = evt_split[-1]
        assert lagstr[-1] == '}', f"syntax incomplete: {event}"
        lagstr = lagstr[:-1]
        lags = [(int(t[1:]) if len(t) > 1 else 0) for t in lagstr.split(",")]
        return lags
    else:
        return [0]


def trial_vector_time_lag(vec, t):
    if t == 0:
        return vec
    dtype = vec.dtype
    if np.issubdtype(dtype, np.bool_):
        oarr = np.zeros(len(vec), dtype=dtype)
    else:
        oarr = np.full(len(vec), np.nan, dtype=dtype)
    if t < 0:
        oarr[-t:] = vec[:t]
    else:
        oarr[:-t] = vec[t:]
    return oarr


def get_behavior_times(mat, behavior):
    """ Takes in behavior{t-k} or behavior,
    :param mat:
    :param behavior: str for behavior events, use {t-k} to zoom in time lags
    :param lag:
    :return: (s x K) where s is determined by lag or behavior arguments
    """
    behavior = behavior.replace(" ", "")
    lags = event_parse_lags(behavior)
    behavior = behavior.split("{")[0]

    if behavior == 'outcome':
        variables = ["contra_rew", "contra_unrew", "ipsi_rew", "ipsi_unrew"]
    elif behavior == 'choice':
        variables = ["left_in_choice", "right_in_choice"]
    elif behavior == 'side_out' or behavior == 'initiate':
        variables = ['initiate']
    elif behavior == 'center_out' or behavior == 'execute':
        variables = ['execute']
    elif behavior == "center_in":
        variables = ['center_in']
    else:
        raise NotImplementedError(f"Unknown behavior {behavior}")
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    k = get_trial_num(mat)
    behavior_times = np.full(k, np.nan)
    for v in variables:
        trials = access_mat_with_path(mat, f"trials/{v}", ravel=True, dtype=np.int)
        times = access_mat_with_path(mat, f"time/{v}", ravel=True)
        behavior_times[trials - 1] = times

    behavior_times = np.vstack([trial_vector_time_lag(behavior_times, l) for l in lags])
    return behavior_times








