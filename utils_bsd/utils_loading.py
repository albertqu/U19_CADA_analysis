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
from utils_bsd.configs import CACHE_FOLDER, DATA_ROOT, VERSION, PLOT_FOLDER


def load_neural_data_BSD(
    sn_cutoff=14, drop_subj=None, return_pse=True, raw=False, **kwargs
):
    data_root = DATA_ROOT
    cache_folder = CACHE_FOLDER
    PS_Expr.info_name = "probswitch_neural_subset_BSD.csv"
    pse = PS_Expr(data_root, modeling_id=None)

    sessions = {
        "BSD011": ["p151", "p154, " "p155", "p156", "p157", "p184"],
        "BSD013": ["p151", "p152", "p153", "p157", "p158", "p184"],
        "BSD015": ["p104", "p105", "p107", "p108", "p111", "p117", "p119"],
        "BSD016": [
            "p106",
            "p107",
            "p109",
            "p110",
            "p112",
            "p113",
            "p122",
            "p124",
            "p125",
        ],
        "BSD017": [
            "p153",
            "p154",
            "p156",
            "p157",
            "p159",
            "p160",
            "p162",
            "p163",
            "p166",
        ],
        "BSD018": [
            "p147",
            "p148",
            "p150",
            "p151",
            "p153",
            "p154",
            "p157",
            "p158",
            "p160",
            "p161",
            "p163",
            "p166",
            "p167",
        ],
        "BSD019": [
            "p139",
            "p140",
            "p142",
            "p143",
            "p145",
            "p146",
            "p148",
            "p150",
            "p152",
            "p153",
            "p155",
            "p156",
            "p158",
        ],
    }

    animal_list_parquet = oj(cache_folder, f"BSD11-19_albert_nb_df_None.pq")
    if not os.path.exists(animal_list_parquet):
        nbdfs = []
        for animal in sessions:
            nb_df_i = pse.align_lagged_view(
                "BSD",
                ["last_side_out", "center_in", "outcome"],
                animal=animal,
                session=lambda s: s.isin(sessions[animal]),
            )
            nbdfs.append(nb_df_i)
        nb_df = pd.concat(nbdfs, axis=0)
        nb_df.loc[
            (nb_df["action"] == nb_df["hemi"]) & (~nb_df["action"].isnull()),
            "ego_action",
        ] = "ipsi"
        nb_df.loc[pds_neq(nb_df["action"], nb_df["hemi"]), "ego_action"] = "contra"
        nb_df = pse.nbm.get_reward_switch_number(nb_df)
        nb_df = pse.nbm.get_OHSH(nb_df)
        nb_df["exec_time"] = nb_df["center_out"] - nb_df["last_side_out"].shift(1)
        nb_df.loc[nb_df["trial"] == 1, "exec_time"] = np.nan
        nb_df["ITI"] = nb_df["center_in"] - nb_df["first_side_out"].shift(1)
        nb_df.loc[nb_df["trial"] == 1, "ITI"] = np.nan
        nb_df["port_dur"] = nb_df["first_side_out"] - nb_df["outcome"]
        nb_df.to_parquet(animal_list_parquet)
    else:
        nb_df = pd.read_parquet(animal_list_parquet)
        pse.nbm.nb_cols, pse.nbm.nb_lag_cols = pse.nbm.parse_nb_cols(nb_df)
    nb_df["center_dur"] = nb_df["center_out"] - nb_df["center_in"]
    nb_df["MVMT"] = nb_df["center_in"] - nb_df["last_side_out"].shift(1)
    nb_df.loc[nb_df["trial"] == 1, "MVMT"] = np.nan
    nb_df[["center_in{t+1}", "center_out{t+1}"]] = nb_df[
        ["center_in", "center_out"]
    ].shift(-1)
    next_session_sel = nb_df["trial"].shift(-1) == 1
    nb_df.loc[next_session_sel, ["center_in{t+1}", "center_out{t+1}"]] = np.nan
    nb_df["SITI{t+1}"] = nb_df["center_out{t+1}"] - nb_df["outcome"]

    pse.nbm.RAND = 230
    # TODO: add meta data
    pse.nbm.sn_cutoff = sn_cutoff
    if drop_subj is not None:
        nb_df = nb_df[~nb_df["animal"].isin(drop_subj)].reset_index(drop=True)
    nb_df = nb_df[(nb_df["session_num"] <= pse.nbm.sn_cutoff)].reset_index(drop=True)
    if not raw:
        neur_df = add_neur_summary(nb_df, pse, **kwargs)
    else:
        neur_df = nb_df
    neur_df["session_num"] = neur_df["session_num"].astype(int)
    neur_df["SO_0"] = neur_df["zeroth_side_out"] - neur_df["outcome"]
    neur_df["SO_lat01"] = neur_df["first_side_out"] - neur_df["zeroth_side_out"]
    neur_df["CIOt"] = neur_df["center_in{t+1}"] - neur_df["outcome"]
    neur_df["prev_reward"] = neur_df["rewarded"].shift(1)
    neur_df.loc[neur_df["trial"] <= 1, "prev_reward"] = np.nan
    neur_df["prev_action"] = neur_df["action"].shift(1)
    neur_df.loc[neur_df["trial"] <= 1, "prev_action"] = np.nan

    if return_pse:
        return neur_df, pse
    else:
        return neur_df


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
        data = add_trial_history(data, length=3, inplace=True)
        data = add_trial_history(data, length=4, inplace=True)
    if drop_subj is not None:
        data = data[~data["Subject"].isin(drop_subj)].reset_index(drop=True)
    return data


def extract_metadata(nb_df_behavior, drop_subj=None, sn_cutoff=14):
    meta_cols = [
        "animal",
        "session_num",
        "session",
        "age",
        "pre_task_weight_percent",
        "post_task_weight_percent",
        "duration",
        "time_in",
        "weightP0",
        "trial2sw",
        "left_score",
        "right_score",
        "hemi",
        "plug_in",
        "recorded",
        "sex",
        "cell_type",
        "left_region",
        "left_virus",
        "right_region",
        "right_virus",
    ]

    nb_df_meta = nb_df_behavior[meta_cols].drop_duplicates().reset_index(drop=True)
    nb_df_meta = nb_df_meta[nb_df_meta["session_num"] <= sn_cutoff].reset_index(
        drop=True
    )
    if drop_subj is not None:
        nb_df_meta = nb_df_meta[~(nb_df_meta["animal"].isin(drop_subj))].reset_index(
            drop=True
        )

    dur = pd.to_datetime(nb_df_meta["duration"], format="%H:%M")
    nb_df_meta["duration_m"] = dur.dt.hour * 60 + dur.dt.minute
    t_in = pd.to_datetime(nb_df_meta["time_in"], format="%H:%M")
    nb_df_meta["time_in_h"] = t_in.dt.hour + t_in.dt.minute / 60
    nb_df_meta["time_in"] = t_in.dt.time
    nb_df_meta["weight_gain"] = (
        nb_df_meta["post_task_weight_percent"] - nb_df_meta["pre_task_weight_percent"]
    )
    sms = ["right_score", "left_score"]
    for m in sms:
        nb_df_meta.loc[nb_df_meta[m] > 1, m] = (
            nb_df_meta.loc[nb_df_meta[m] > 1, m] / 100
        )
    nb_df_meta["total_score"] = (
        nb_df_meta["left_score"] + nb_df_meta["right_score"]
    ) / 2

    nb_df_meta["right_bias"] = nb_df_meta["right_score"] - nb_df_meta["left_score"]
    nb_df_meta["recorded"].fillna(0, inplace=True)
    nb_df_meta["hemi"].replace({"": "NA"}, inplace=True)
    return nb_df_meta


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
        data.loc[uusel & consec_stay & after_start, "stayHistory"] = "UU"
        data.loc[ursel & consec_stay & after_start, "stayHistory"] = "UR"
        data.loc[rusel & consec_stay & after_start, "stayHistory"] = "RU"
        data.loc[rrsel & consec_stay & after_start, "stayHistory"] = "RR"
        data["stayHistory"] = pd.Categorical(
            data["stayHistory"], categories=["RR", "UR", "RU", "UU"], ordered=True
        )
        return data


# implement celia's history method
def add_trial_history(data, length=3, inplace=False):
    # use `~df['history'].str.contains('N')` to filter out bad columns
    hist_arg = "history" if length == 3 else f"history{length}"
    if inplace:
        df = data
    else:
        df = data.copy()
    # length = 3
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
    df[hist_arg] = np.add.reduce([df[l] for l in lcols[::-1]])
    df.drop(columns=ccols + rcols + lcols, inplace=True)
    mean_switch = (
        df.loc[~df[hist_arg].str.contains("N"), [hist_arg, "Switch"]]
        .groupby([hist_arg], as_index=False)
        .agg("mean")
    )
    mean_switch.rename(columns={"Switch": f"{hist_arg}_avg_switch"}, inplace=True)
    df = pd.merge(df, mean_switch, on=hist_arg, how="left")
    return df


def load_model_data(
    data_arg, model_strs, version=VERSION, sn_cutoff=14, drop_subj=None
):
    # RPE include
    all_mdf = []
    # for model in [PCModel_fixpswgam, BIModel_fixp, RLCF, RL_4p, PCModel, BIModel, LR]:
    for model_str in model_strs:
        model = load_model(model_str)
        mdl = model()
        model_arg = str(mdl)
        pfile = os.path.join(
            CACHE_FOLDER, f"bsd_simopt_data_{data_arg}_{model_arg}_{version}.csv"
        )
        # .values keep reference to original dataframe!!
        if model == LR:
            mdf = pd.read_csv(pfile)[mdl.data_cols + ["choice_p"]].copy()
            action_prob = mdf["choice_p"].values.copy()
            action_prob[mdf["Decision"] == -1] = 0
            action_prob[mdf["Decision"] == 0] = 1 - action_prob[mdf["Decision"] == 0]
            mdf["rpe"] = mdf["Reward"] - action_prob
        elif model == RFLR:
            mdf = pd.read_csv(pfile)[mdl.data_cols + ["choice_p"]].copy()
            mdf["rpe"] = np.nan
        else:
            mdf = pd.read_csv(pfile)[mdl.data_cols + ["rpe", "choice_p"]].copy()
        mdf.loc[mdf["Target"] == 1, "correct"] = mdf["choice_p"]
        mdf.loc[mdf["Target"] == 0, "correct"] = 1 - mdf["choice_p"]
        prev_choice = mdf["Decision"].shift(1)
        mdf["stay%"] = 0.0
        mdf["Switch"] = 0.0
        mdf.loc[prev_choice == 1, "stay%"] = mdf["choice_p"]
        mdf.loc[prev_choice == 0, "stay%"] = 1 - mdf["choice_p"]
        mdf.loc[prev_choice == 1, "Switch"] = 1 - mdf["choice_p"]
        mdf.loc[prev_choice == 0, "Switch"] = mdf["choice_p"]
        mdf.loc[mdf["Trial"] == 1, "stay%"] = 0
        mdf.loc[mdf["Trial"] == 1, "Switch"] = 0
        mdf["source"] = model_arg
        mdf["session_num"] = mdf["Session"].apply(lambda s: int(s.split("_")[1]))
        mdf_sel = mdf["session_num"] <= sn_cutoff
        if drop_subj is not None:
            mdf_sel = mdf_sel & (~mdf["Subject"].isin(drop_subj))
        all_mdf.append(mdf[mdf_sel])
    all_mdf = pd.concat(all_mdf, axis=0).reset_index(drop=True)
    return all_mdf


def add_neur_summary(nb_df, pse, event="outcome", debase=True, tstart=0.3, tend=1.301):
    # TODO: PUT IN UTILS
    col_sels = [
        "animal",
        "session",
        "roi",
        "state",
        "trial",
        "session_num",
        "switch_num",
        "reward_num",
        "ego_action",
        "rewarded",
        "quality",
        "OH",
        "SH",
        "trial_in_block",
        "hemi",
        "action",
        "exec_time",
        "MVMT",
        "center_dur",
        "ITI",
        "first_side_out",
        "center_in",
        "center_out",
        "outcome",
        "last_side_out",
        "zeroth_side_out",
        "port_dur",
        "SITI{t+1}",
        "center_out{t+1}",
        "center_in{t+1}",
    ]
    df = nb_df[col_sels + pse.nbm.nb_cols[pse.nbm.default_ev_neur(event)]].reset_index(
        drop=True
    )
    nb_df = df
    # Step 2: dim reduce data and append mean/PC -> no more lagging operation available after
    dropcols_pca = [
        c
        for c in pse.nbm.nb_cols[pse.nbm.default_ev_neur(event)]
        if (not pse.nbm.align_time_in(c, tstart, tend))
    ]
    if debase:
        targ_cols = pse.nbm.nb_cols[event + "_neur"]
        zero_col = [c for c in targ_cols if (float(c.split("|")[1]) == 0)][0]
        df[targ_cols] = df[targ_cols].values - df[zero_col].values[:, np.newaxis]
    df = df.dropna(subset=pse.nbm.nb_cols[pse.nbm.default_ev_neur(event)]).reset_index(
        drop=True
    )
    # todo; throw away abort/omit (no action) trials
    dim_red1 = pse.nbm.apply_to_idgroups(
        df.drop(columns=dropcols_pca),
        NBM_Preprocessor(pse.nbm).neural_dim_reduction,
        event="outcome",
        method=3,
    ).reset_index(drop=True)
    dim_red2 = pse.nbm.apply_to_idgroups(
        df.drop(columns=dropcols_pca),
        NBM_Preprocessor(pse.nbm).neural_dim_reduction,
        event="outcome",
        method="mean",
    ).reset_index(drop=True)
    df = (
        pd.concat([df[["animal", "session_num", "trial"]], dim_red1, dim_red2], axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    cname = [c for c in df.columns if "_mean(" in c][0]
    df.rename(columns={cname: "outcome_DA_mean"}, inplace=True)
    neur_df = nb_df[col_sels + pse.nbm.nb_cols[pse.nbm.default_ev_neur(event)]].merge(
        df, on=["animal", "session_num", "trial"], how="left"
    )
    print("extracting", cname)
    return neur_df


def model2neural(nb_df, mdlr_df, data_df):
    # Switch, rpe, choice_p, stay%, source, session_num, Subject, Trial
    v = pd.pivot_table(
        mdlr_df,
        values=["choice_p", "rpe", "Switch", "stay%"],
        index=["Subject", "session_num", "Trial"],
        columns="source",
    )
    new_cols = [f"{p1}__{p2}" for p1, p2 in v.columns]
    v.columns = new_cols
    v.reset_index(inplace=True)
    v = v.merge(
        data_df[
            [
                "Subject",
                "session_num",
                "Trial",
                "history",
                "history_avg_switch",
                "Switch",
                "stayHistory",
            ]
        ],
        how="left",
        on=["Subject", "session_num", "Trial"],
    )
    nb_df = nb_df.merge(
        v,
        how="left",
        left_on=["animal", "session_num", "trial"],
        right_on=["Subject", "session_num", "Trial"],
    )
    return nb_df


def peak_trough_smart(df, neurcols, t_start, t_end, offset_col=None):
    data = df[neurcols].values
    xts = np.array([float(c.split("|")[1]) for c in neurcols])
    port_dur = df["port_dur"].values
    center_in_next = df["center_in{t+1}"].values
    rewarded = df["rewarded"].values
    if offset_col is not None:
        offsets = df[offset_col].values
    else:
        offsets = np.zeros(len(data))

    def extrema(x, offset=0):
        if np.all(np.isnan(x)):
            return np.nan
        maxi, mini = np.nanmax(x) - offset, np.nanmin(x) - offset
        if abs(maxi) > abs(mini):
            return maxi
        else:
            return mini

    arr = np.zeros(len(data))
    for i in range(len(data)):
        if pd.isnull(rewarded[i]):
            arr[i] = np.nan
        elif rewarded[i]:
            upper = port_dur[i]
        else:
            upper = center_in_next[i]
        if pd.isnull(upper):
            arr[i] = np.nan
        else:
            d = data[i, (xts >= t_start) & (xts <= min(upper, t_end))]
            arr[i] = extrema(d, offsets[i])
    return arr


def peak_trough_outcome(
    df, neurcols, t_start, t_end, rew_event, unrew_event, offset_col=None
):
    data = df[neurcols].values
    xts = np.array([float(c.split("|")[1]) for c in neurcols])
    rcut = df[rew_event].values
    ucut = df[unrew_event].values
    rewarded = df["rewarded"].values
    if offset_col is not None:
        offsets = df[offset_col].values
    else:
        offsets = np.zeros(len(data))
    arr = np.zeros(len(data))
    for i in range(len(data)):
        if pd.isnull(rewarded[i]):
            arr[i] = np.nan
            continue
        elif rewarded[i]:
            upper = rcut[i]
        else:
            upper = ucut[i]
        if pd.isnull(upper):
            arr[i] = np.nan
        else:
            d = data[i, (xts >= t_start) & (xts <= min(upper, t_end))]
            arr[i] = max(d) if rewarded[i] else min(d)
    return arr
