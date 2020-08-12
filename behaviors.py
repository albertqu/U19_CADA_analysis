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
        mat = access_mat_with_path(mat, "glml")
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


# TODO: take into account of possibility of duplicates
def get_trial_outcome_laterality(mat):
    """
    :param mat:
    :return: ipsi, contra trials respectively
    """
    results = {}
    for side in 'ipsi', 'contra':
        rew = access_mat_with_path(mat, f'glml/trials/{side}_rew', ravel=True, dtype=np.int)
        unrew = access_mat_with_path(mat, f'glml/trials/{side}_unrew', ravel=True, dtype=np.int)
        results[side] = np.sort(np.concatenate(rew, unrew))
    return results


def get_trial_outcomes(mat, as_array=True):
    """
    :param mat:
    :param as_array: if True, returns array instead of dict
    :return: rewarded, unrewarded trials
    Not using boolean due to possibility of an no-outcome trial
    """
    results = {}
    if as_array:
        raise NotImplementedError()
    else:
        for outcome in 'rew', 'unrew':
            ipsi = access_mat_with_path(mat, f'glml/trials/ipsi_{outcome}', ravel=True, dtype=np.int)
            contra= access_mat_with_path(mat, f'glml/trials/contra_{outcome}', ravel=True, dtype=np.int)
            results[outcome+'arded'] = np.sort(np.concatenate(ipsi, contra))
        return results


def get_outcome_times(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml")
    k = np.prod(access_mat_with_path(mat, 'trials/ITI').shape)
    variables = ["contra_rew", "contra_unrew", "ipsi_rew", "ipsi_unrew"]
    outcome_times = np.full(k, np.nan)
    for v in variables:
        trials = access_mat_with_path(mat, f"trials/{v}", ravel=True, dtype=np.int)
        times = access_mat_with_path(mat, f"time/{v}", ravel=True)
        outcome_times[trials-1] = times
    return outcome_times








