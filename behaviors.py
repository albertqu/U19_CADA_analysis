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
    outcome_times = get_outcome_times(mat)
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
    return decode_trial_behavior(outcomes, {'No choice': 3, 'Incorrect': 2, 'Correct Omission': 1.1,
                                            'Rewarded': 1.2})


def get_trial_num(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    return np.prod(access_mat_with_path(mat, 'trials/ITI').shape)


def decode_trial_behavior(arr, code):
    return {c: arr == code[c] for c in code}


def get_outcome_times(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    k = np.prod(access_mat_with_path(mat, 'trials/ITI').shape)
    variables = ["contra_rew", "contra_unrew", "ipsi_rew", "ipsi_unrew"]
    outcome_times = np.full(k, np.nan)
    for v in variables:
        trials = access_mat_with_path(mat, f"trials/{v}", ravel=True, dtype=np.int)
        times = access_mat_with_path(mat, f"time/{v}", ravel=True)
        outcome_times[trials-1] = times
    return outcome_times








