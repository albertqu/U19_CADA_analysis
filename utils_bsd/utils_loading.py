import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special

# import utilities
from utils import *
from behaviors import *
from packages.photometry_functions import get_zdFF
from sklearn.linear_model import LogisticRegression
from peristimulus import *
from neuro_series import *
from statsmodels.tsa.tsatools import lagmat
from neurobehavior_base import *
from nb_viz import *
import sklearn
import numbers
import logging
import itertools
from cogmodels import *
import plotly.express as px
import plotly.graph_objects as go
from utils_bsd.configs import CACHE_FOLDER


def load_behavior_data(data_arg, sn_cutoff=11, drop_subj=None, history=True):
    # Should we drop BSD012 if we have no neural data?
    # Decision == 1 -> rightward action, 0 -> leftward action
    cache_folder = CACHE_FOLDER
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    data["correct"] = data["Decision"] == data["Target"]
    data["session_num"] = data["Session"].apply(lambda s: int(s.split("_")[1]))
    data["stay%"] = 1 - data["Switch"]
    nansel = (data["Decision"] == -1) | (data["Decision"].shift(1) == -1)
    data.loc[(data["Trial"] == 1) | nansel, "stay%"] = 0
    if sn_cutoff is not None:
        data = data[data["session_num"] <= sn_cutoff].reset_index(drop=True)
    if history:
        data = add_stayHistory(data)
    if drop_subj is not None:
        data = data[~data["Subject"].isin(drop_subj)].reset_index(drop=True)
    return data


def add_stayHistory(data):
    if "stayHistory" in data.columns:
        return data
    else:
        r1back = data["Reward"].shift(1)
        r2back = data["Reward"].shift(2)
        uusel = (r1back == 0) & (r2back == 0)
        ursel = (r1back == 1) & (r2back == 0)
        rusel = (r1back == 0) & (r2back == 1)
        rrsel = (r1back == 1) & (r2back == 1)
        consec_stay = data["Decision"].shift(1) == data["Decision"].shift(2)
        after_start = data["Trial"] > 2
        data.loc[uusel & consec_stay & after_start, "stayHistory"] = "--"
        data.loc[ursel & consec_stay & after_start, "stayHistory"] = "-+"
        data.loc[rusel & consec_stay & after_start, "stayHistory"] = "+-"
        data.loc[rrsel & consec_stay & after_start, "stayHistory"] = "++"
        data["stayHistory"] = pd.Categorical(
            data["stayHistory"], categories=["++", "-+", "+-", "--"], ordered=True
        )
        return data


# implement celia's history method
def add_trial_history(data, length=3, inplace=False):
    # use `~df['history'].str.contains('N')` to filter out bad columns
    if inplace:
        df = data
    else:
        df = data.copy()
    length = 3
    cs = lagmat(df["Decision"], maxlag=length, trim="forward", original="ex")
    rs = lagmat(df["Reward"], maxlag=length, trim="forward", original="ex")
    # initialize everything as N, make the farthest column a, and then fill in the rest
    ccols = [f"C{i}" for i in range(1, length + 1)]
    rcols = [f"R{i}" for i in range(1, length + 1)]
    lcols = [f"L{i}" for i in range(1, length + 1)]
    df[ccols] = cs
    df[rcols] = rs
    df[lcols] = "N"
    df.loc[cs[:, length - 1] != -1, f"L{length}"] = "a"
    for i in range(length - 1):
        vsel = cs[:, i] != -1
        df.loc[(cs[:, i] == cs[:, -1]) & vsel, f"L{i+1}"] = "a"
        df.loc[(cs[:, i] != cs[:, -1]) & vsel, f"L{i+1}"] = "b"
    for i in range(length):
        df.loc[rs[:, i] == 1, f"L{i+1}"] = df.loc[rs[:, i] == 1, f"L{i+1}"].str.upper()
    df.loc[df["Trial"] <= length, lcols] = "N"
    df["history"] = np.add.reduce([df[l] for l in lcols[::-1]])
    df.drop(columns=ccols + rcols + lcols, inplace=True)
    mean_switch = (
        df.loc[~df["history"].str.contains("N"), ["history", "Switch"]]
        .groupby(["history"], as_index=False)
        .agg("mean")
    )
    mean_switch.rename(columns={"Switch": "history_avg_switch"}, inplace=True)
    df = pd.merge(df, mean_switch, on="history", how="left")
    return df
