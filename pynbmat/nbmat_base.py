import numpy as np
from neuro_series import *
from nb_viz import *
from peristimulus import *
from abc import abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.tsa.tsatools import lagmat
from utils import decode_from_regfeature
from utils_system import find_files_recursive
import logging
import patsy
import sklearn
from tqdm import tqdm
import traceback

logging.basicConfig(level=logging.INFO)
# sns.set_context("talk")
RAND_STATE = 230


##################################################
##################### NBMat ######################
##################################################
class NeuroBehaviorMat:
    # always be careful of applying lagging operations or dim reductions and check if:
    # 1. operations are only carried over rows with the same (animal, session) tuple
    # 2. df is in wide format, where data is organized trial by trials
    # TODO: event selection tutorial

    behavior_events = []
    event_features = {}
    trial_features = []
    id_vars = []
    time_multiplier = 1

    id_vars = ["animal", "session", "roi"]
    uniq_cols = id_vars + ["trial"]

    # example id_vars: animal, session, trial,
    # make a state representation?
    def __init__(self, neural=True, expr=None):
        self.expr = expr
        self.neural = neural
        # self.id_vars = id_vars
        self.event_time_windows = {}
        self.nb_cols = None
        self.nb_lag_cols = None

    @staticmethod
    def align_time_in(s, start, end, include_upper=False):
        t = float(s.split("|")[1])
        if include_upper:
            return (t >= start) & (t <= end)
        return (t >= start) & (t < end)

    def default_ev_neur(self, ev):
        if "{" in ev:
            ev1, lag = ev.split("{")
            return f"{ev1}_neur" + "{" + lag
        else:
            return ev + "_neur"

    def parse_nb_cols(self, nb_df, ev_neur_func=None):
        # assume roi_long form
        if ev_neur_func is None:
            ev_neur = self.default_ev_neur
        else:
            ev_neur = ev_neur_func
        lag_form = r"([-|\w]+)_neur({t-\d})|.*"
        std_form = r"([-|\w]+)_neur|.*"
        all_nbcols = [c for c in nb_df.columns if ("_neur" in c) and ("|" in c)]
        nb_lag_cols = {}
        nb_cols = {}
        ignore_roi = lambda s: s.split("--")[1] if ("--" in s) else s
        for c in all_nbcols:
            lagm = re.match(lag_form, c)
            stdm = re.match(std_form, c)
            # print(c, lagm, stdm)
            if r"{" in c:
                evt, lag = lagm.group(1), lagm.group(2)[1:-1]
                if ev_neur(evt) in nb_lag_cols:
                    if lag in nb_lag_cols[ev_neur(evt)]:
                        nb_lag_cols[ev_neur(evt)][lag].append(c)
                    else:
                        nb_lag_cols[ev_neur(evt)][lag] = [c]
                else:
                    nb_lag_cols[ev_neur(evt)] = {lag: [c]}
            elif stdm:
                evt = stdm.group(1)
                if ev_neur(evt) in nb_cols:
                    nb_cols[ev_neur(evt)].append(c)
                else:
                    nb_cols[ev_neur(evt)] = [c]
        return nb_cols, nb_lag_cols

    def series_time(self, event, t):
        ev_neur = self.default_ev_neur
        return f"{ev_neur(event)}|{t * self.time_multiplier :.2f}"

    def extend_features(self, nb_df, *args, **kwargs):
        return nb_df

    def merge_rois_wide(self, df, pivot_col):
        # nb_df must be of wide form: meaning each trial must only have 1
        id_cols = self.id_vars
        ind_ics = list(np.setdiff1d(id_cols, [pivot_col]))
        return self.apply_to_idgroups(
            df, self.merge_rois_wide_ID, id_vars=ind_ics, pivot_col=pivot_col
        )

    def merge_rois_wide_ID(self, df, pivot_col):
        id_cols = self.id_vars

        ind_ics = list(np.setdiff1d(id_cols, [pivot_col]))
        nb_cols = [c for c in df.columns if ("_neur" in c) and ("|" in c)]
        pivot_values = df[pivot_col].unique()
        # print(ind_ics, pivot_values)

        result_df = df.loc[df[pivot_col] == pivot_values[0]].drop(columns=pivot_col)
        result_df.rename(
            columns={nbc: f"{pivot_values[0]}--{nbc}" for nbc in nb_cols}, inplace=True
        )
        for i, pv in enumerate(pivot_values):
            slice_df = df.loc[df[pivot_col] == pv, ind_ics + nb_cols].reset_index(
                drop=True
            )
            slice_df.rename(
                columns={nbc: f"{pv}--{nbc}" for nbc in nb_cols}, inplace=True
            )
            result_df = result_df.merge(slice_df, how="left", on=ind_ics)

    def align_B2N_dff_ID(self, behavior_df, neur_df, events, form="wide"):
        # align behavior to neural on a single session basis, DO NOT use it across multiple sessions
        # event_windows: dictionary with event to window series
        # Now default to aligning to dff, assumes that df is already in tidy format,
        # for raw fluorescence should first melt then compare

        """
        TODO: check edge case when neural_df ends right around when behavior ends, making side_out + 2s impossible
        by default takes in neur_df
        Now method only supports alignment for single animal session, need to develop ID system
        Returns:
            aligned_df: dataframe with behavior features
            nb_cols: column names of the new aligned cols
        """
        id_set = np.setdiff1d(self.id_vars, ["roi"])
        assert np.all(
            [len(np.unique(behavior_df[v])) == 1 for v in id_set]
        ), "not unique"
        nb_dfs = []
        dff_cname = "ZdFF"
        result_dfs = []
        ev_neur = lambda ev: ev + "_neur"
        nb_cols = {
            ev_neur(ev): [
                f"{ev_neur(ev)}|{ts * self.time_multiplier :.2f}"
                for ts in self.event_time_windows[ev]
            ]
            for ev in events
        }
        for roi in np.unique(neur_df["roi"]):
            ev_nb_df = behavior_df.copy()
            roi_sig = neur_df.loc[neur_df["roi"] == roi, dff_cname].values
            roi_time = neur_df.loc[neur_df["roi"] == roi, "time"].values
            for event in events:
                if event == "side_out":
                    event = "first_side_out"
                assert event in self.behavior_events, f"unknown event {event}"
                ev_sel = ~np.isnan(
                    behavior_df[event]
                )  # TODO: maybe use pd.isnull instead
                aROI = align_activities_with_event(
                    roi_sig,
                    roi_time,
                    behavior_df.loc[ev_sel, event].values,
                    self.event_time_windows[event],
                )
                ev_nb_df.loc[ev_sel, "roi"] = roi
                ev_nb_df.loc[ev_sel, nb_cols[ev_neur(event)]] = aROI

            nb_dfs.append(ev_nb_df)
        nb_df = pd.concat(nb_dfs, axis=0)
        if form == "long":
            all_evneur_cols = np.concatenate(list(nb_cols.values()))
            id_cols = list(np.setdiff1d(nb_df.columns, all_evneur_cols))
            # recursively reduce event column groups
            for event in events:
                ecols = nb_cols[ev_neur(event)]
                nb_df = pd.melt(
                    nb_df,
                    id_vars=id_cols,
                    value_vars=ecols,
                    var_name=f"{ev_neur(event)}_time",
                    value_name=f"{ev_neur(event)}_{dff_cname}",
                )
                nb_df[f"{ev_neur(event)}_time"] = (
                    nb_df[f"{ev_neur(event)}_time"]
                    .str.extract(f"{ev_neur(event)}" + "\|(?P<ts>-?(\d|\.)+)")
                    .ts
                )
                id_cols = id_cols + [
                    f"{ev_neur(event)}_time",
                    f"{ev_neur(event)}_{dff_cname}",
                ]
                nb_df[f"{ev_neur(event)}_time"] = nb_df[
                    f"{ev_neur(event)}_time"
                ].astype(float)
        self.nb_cols = nb_cols
        return nb_df

    def lag_wide_df_ID(self, nb_df, features):
        # lagging option differs in behavior for trial/event features and neural features
        # features = {feature: {'long': T/F (default F), 'pre': int (default 0), 'post': int (default 0)}}
        # TODO: debug
        assert np.all(
            [len(np.unique(nb_df[v])) == 1 for v in self.id_vars]
        ), "not unique"
        assert len(nb_df.trial) == len(
            np.unique(nb_df.trial)
        ), "suspect data is not in wide form"
        for feat in features:
            feat_opt = features[feat]
            flex_t = lambda i, mode: "{t-%d}" % i if mode == "pre" else "{t+%d}" % i

            def te_colf(fts, lag, mode):
                if len(fts) > 1:
                    logging.warning("unexpected")
                ft = fts[0]
                assert mode in ["pre", "post"], f"Bad Mode {mode}"
                return [ft + flex_t(i, mode) for i in range(1, lag + 1)]

            def neur_colf(fts, lag, mode):
                assert mode in ["pre", "post"], f"Bad Mode {mode}"
                sif = (
                    lambda x, i, mode: x.split("|")[0]
                    + flex_t(i, mode)
                    + "|%s" % x.split("|")[1]
                )
                return np.concatenate(
                    [[sif(ft, i, mode) for ft in fts] for i in range(1, lag + 1)]
                )

            if "_neur" in feat:
                feat_ev = "_".join(feat.split("_")[:-1])
                assert feat in self.nb_cols, f"event selected {feat_ev} not aligned yet"
                cols_to_shifts = self.nb_cols[feat]
                colf = neur_colf
            else:
                if not (
                    (feat in self.trial_features + self.behavior_events)
                    or (feat in self.event_features)
                ):
                    # print(f'Lagging derived feature {feat}, can lead to unexpected behavior')
                    assert feat in nb_df.columns, f"unknown option {feat}"
                cols_to_shifts = [feat]
                colf = te_colf
            values_to_shifts = nb_df[cols_to_shifts]
            shifted = []
            shifted_cols = []
            # add method for interval shifting if needed
            if "pre" in feat_opt and feat_opt["pre"]:
                # lagmat pads with 0 use df.shift instead
                shifted.append(
                    np.concatenate(
                        [
                            values_to_shifts.shift(i)
                            for i in range(1, feat_opt["pre"] + 1)
                        ],
                        axis=1,
                    )
                )
                # add shifted_cols
                shifted_cols.append(colf(cols_to_shifts, feat_opt["pre"], "pre"))

            if "post" in feat_opt and feat_opt["post"]:
                shifted.append(
                    np.concatenate(
                        [
                            values_to_shifts.shift(-i)
                            for i in range(1, feat_opt["post"] + 1)
                        ],
                        axis=1,
                    )
                )
                shifted_cols.append(colf(cols_to_shifts, feat_opt["post"], "post"))
            if shifted:
                assert shifted, "didnt do anything?"
                shifted_cols = list(np.concatenate(shifted_cols))
                nb_df[shifted_cols] = np.concatenate(shifted, axis=1)
            if "long" in feat_opt and feat_opt["long"]:
                features[feat]["to_melt"] = cols_to_shifts + shifted_cols
        for feat in features:
            # check melt: TODO: currently assume all event_neur are lagged
            # update lagged cols
            if "long" in features[feat] and features[feat]["long"]:
                melt_cols = features[feat]["to_melt"]
                id_cols = np.setdiff1d(nb_df.columns, melt_cols)
                # assert '_neur' in feat, 'currently only support aligned neural'
                if "_neur" in feat:
                    # TODO: flexible _ZdFF treatment
                    nb_df = nb_df.melt(
                        id_vars=id_cols,
                        value_vars=melt_cols,
                        var_name=f"{feat}_arg",
                        value_name=f"{feat}_ZdFF",
                    )
                    nb_df[[f"{feat}_lag", f"{feat}_time"]] = (
                        nb_df[f"{feat}_arg"]
                        .str.replace(feat, "")
                        .str.split(r"|", expand=True)
                    )
                    nb_df[f"{feat}_time"] = nb_df[f"{feat}_time"].astype(float)
                else:
                    nb_df = nb_df.melt(
                        id_vars=id_cols,
                        value_vars=melt_cols,
                        var_name=f"{feat}_arg",
                        value_name=f"{feat}_value",
                    )
                    sample_val = nb_df.loc[0, f"{feat}_arg"]
                    if "|" in sample_val:
                        nb_df[[f"{feat}_lag", f"{feat}_time"]] = (
                            nb_df[f"{feat}_arg"]
                            .str.replace(feat, "")
                            .str.split(r"|", expand=True)
                        )
                        nb_df[f"{feat}_time"] = nb_df[f"{feat}_time"].astype(float)
                    else:
                        nb_df[f"{feat}_lag"] = nb_df[f"{feat}_arg"].str.replace(
                            feat, ""
                        )
                nb_df[f"{feat}_lag"] = nb_df[f"{feat}_lag"].apply(
                    lambda x: x[1:-1].replace("t", "")
                )
                nb_df.loc[nb_df[f"{feat}_lag"] == "", f"{feat}_lag"] = 0
                nb_df[f"{feat}_lag"] = nb_df[f"{feat}_lag"].astype(int)
                nb_df.drop(columns=f"{feat}_arg", inplace=True)
                uniq_lag = np.unique(nb_df[f"{feat}_lag"])
                if (len(uniq_lag) == 1) and (uniq_lag[0] == 0):
                    nb_df.drop(columns=f"{feat}_lag", inplace=True)
        return nb_df

    def lag_wide_df(self, nb_df, features):
        # nb_df must be of wide form: meaning each trial must only have 1
        return self.apply_to_idgroups(nb_df, self.lag_wide_df_ID, features=features)

    def register_roi_ID(self, nb_df, virus_map=None):
        default_virus_map = {"jRGECO1a": "red", "dLight1.3b": "green"}
        if virus_map is None:
            virus_map = default_virus_map
        # TODO: currently only works with Chris Probswitch
        assert np.all(
            [len(np.unique(nb_df[v])) == 1 for v in self.id_vars]
        ), "not unique"
        assert "roi" in self.id_vars, "nb_df must have roi"
        roi = nb_df.loc[0, "roi"]
        roi_splits = roi.split("_")
        assert len(roi_splits) > 1, "register ID only works with full ROIs"
        if len(roi_splits) == 2:
            roi_splits = [nb_df.loc[0, "hemi"]] + roi_splits
        roi_opt = roi_splits[0]
        color_opt = roi_splits[1]
        virus_args = nb_df.loc[0, f"{roi_opt}_virus"].split("/")
        virus = ""
        for varg in virus_args:
            if virus_map[varg] == color_opt:
                virus = varg
        region = nb_df.loc[0, f"{roi_opt}_region"]
        nb_df["virus"] = virus
        nb_df["region"] = region
        return nb_df

    def register_roi(self, nb_df, virus_map=None):
        return self.apply_to_idgroups(nb_df, self.register_roi_ID, virus_map=virus_map)

    def apply_to_idgroups(self, nb_df, func, id_vars=None, *args, **kwargs):
        """func: must takes in nb_df,
        *args, **kwargs: additional argument for func
        potentially replace with:
        import pandas as pd
        df = pd.DataFrame({'A': [2, 2, 3, 3, 4, 4, 4], 'B': [2, 2, 2,2, 3, 3, 3], 'C': np.arange(7)})
        def func(x):
            print(x)
            return pd.DataFrame({'cs': [x['C'].mean(), x['C'].std(), x['C'].mean()**2], 'ctype': ['mean', 'std', 'quad']})

        df.groupby(["A","B"]).apply(func).droplevel(2).reset_index()
        """
        #
        if id_vars is None:
            id_vars = self.id_vars
        # id_groups = [np.unique(nb_df[v]) for v in id_vars]
        all_dfs = []
        # better runtime with
        nb_df["combined"] = nb_df[id_vars].values.tolist()
        nb_df["combined"] = nb_df["combined"].str.join(",")
        for id_group in np.unique(nb_df["combined"].dropna()):
            id_group_vars = id_group.split(",")
            uniq_sel = np.logical_and.reduce(
                [nb_df[idv] == id_group_vars[i] for i, idv in enumerate(id_vars)]
            )
            all_dfs.append(
                func(nb_df[uniq_sel].reset_index(drop=True), *args, **kwargs)
            )
        nb_df.drop(columns="combined", inplace=True)
        return pd.concat(all_dfs, axis=0)

    def get_HPM(
        self, nb_df, outcome_col, reward_col, tmax=None, bin_size=5, dropend=False
    ):
        if nb_df[reward_col].dtype != bool:
            nb_df = nb_df.dropna(subset=reward_col)
            nb_df[reward_col] = nb_df[reward_col].astype(bool)
        return self.apply_to_idgroups(
            nb_df,
            self.get_HPM_ID,
            outcome_col=outcome_col,
            reward_col=reward_col,
            tmax=tmax,
            bin_size=bin_size,
            dropend=dropend,
        )

    def get_HPM_ID(
        self, nb_df, outcome_col, reward_col, tmax=None, bin_size=5, dropend=False
    ):
        """

        bin_size: np.float
            time bin size in minutes
        """
        # assuming input is NaN free: nb_df = nb_df.dropna(subset=[outcome_col, reward_col])
        assert nb_df[reward_col].dtype == bool
        assert np.all(
            [len(np.unique(nb_df[v])) == 1 for v in self.id_vars]
        ), "not unique"
        assert len(nb_df.trial) == len(
            np.unique(nb_df.trial)
        ), "suspect data is not in wide form"
        nb_df_tmax = np.max(nb_df[self.behavior_events].values)
        if tmax is None:
            tmax = nb_df_tmax
        # obtain tmax via fp_series: THINK more about the best way
        hits = nb_df.loc[nb_df[reward_col], outcome_col]
        # binning variables
        bigbin = bin_size * 60  # int: seconds
        first_bin_end = bigbin  # frame number: float

        if dropend:
            ebin = tmax + 1
        else:
            ebin = int(np.ceil(tmax / bigbin)) * bigbin + 1
        bins = np.arange(0, ebin, bigbin)  # in seconds
        [hpm, xx] = np.histogram(hits, bins)

        xx = xx[1:]
        if not dropend:
            last_binsize = tmax - bins[-2]
            hpm[-1] *= bigbin / last_binsize
            xx[-1] = last_binsize + xx[-2]

        hpm = hpm / bin_size
        xx = np.around(xx / 60, 2)
        rdf = pd.DataFrame({"time_bin": xx, "HPM": hpm})
        for v in self.id_vars:
            rdf[v] = nb_df[v].unique()[0]
        return rdf

    def get_PC_ID(self, nb_df, reward_col, bin_size=100, drop_end=True):
        perf = (nb_df[reward_col] == True).astype(np.float).values
        n_trial = len(perf)

        if drop_end:
            k = int(np.floor(n_trial / bin_size))
        else:
            k = int(np.ceil(n_trial / bin_size))
            perf = np.concatenate([perf, np.full(k * bin_size - n_trial, np.nan)])

        perf_mat = perf.reshape((k, bin_size), order="C")

        pc = np.nanmean(perf_mat, axis=1)
        xs = np.arange(1, k + 1) * bin_size
        if not drop_end:
            xs[-1] = n_trial
        rdf = pd.DataFrame({"time_bin": xs, "HPM": pc})
        for v in self.id_vars:
            rdf[v] = nb_df[v].unique()[0]

        return rdf

    def dim_reduce_aligned_neural(self, nb_df, event, start=0, end=1.001):
        # find relevant columns, also save dropcols
        dimred_cols = [
            c
            for c in self.nb_cols[self.default_ev_neur(event)]
            if self.align_time_in(c, start, end)
        ]

        # find rows with nonan columns
        nonansels = ~np.any(np.isnan(nb_df[dimred_cols].values), axis=1)
        # subset valid rows, reset index and dim reduce
        data_df = nb_df.loc[nonansels, self.id_vars + dimred_cols].reset_index(
            drop=True
        )

        # save data with nonan

        dim_red1 = self.apply_to_idgroups(
            data_df, NBM_Preprocessor(self).neural_dim_reduction, event=event, method=3
        ).reset_index(drop=True)
        dim_red2 = self.apply_to_idgroups(
            data_df,
            NBM_Preprocessor(self).neural_dim_reduction,
            event=event,
            method="mean",
        ).reset_index(drop=True)
        nb_df.loc[nonansels, list(dim_red1.columns) + list(dim_red2.columns)] = (
            pd.concat([dim_red1, dim_red2], axis=1).values
        )
        return nb_df

    def debase_gradient(self, nb_df, event, base_event=None, b_start=0, b_end=0):
        # t_start, t_end: inclusive
        debase = True
        if debase:
            targ_cols = self.nb_cols[event + "_neur"]
            if base_event is None:
                base_cols = targ_cols
            else:
                base_cols = self.nb_cols[base_event + "_neur"]
            if b_end == b_start:
                zero_col = [
                    c for c in base_cols if (float(c.split("|")[1]) == b_start)
                ][0]
                nb_df[targ_cols] = (
                    nb_df[targ_cols].values - nb_df[zero_col].values[:, np.newaxis]
                )
            else:
                zero_cols = [
                    c
                    for c in base_cols
                    if self.align_time_in(c, b_start, b_end, include_upper=True)
                ][0]
                nb_df[targ_cols] = nb_df[targ_cols].values - np.mean(
                    nb_df[zero_cols].values, axis=1, keepdims=True
                )
        # TODO: get gradient


#########################################################
##################### NBExperiment ######################
#########################################################
class NBExperiment:
    info_name = None
    spec_name = None

    def __init__(self, folder="", modeling_id=None, cache=True):
        self.meta = None
        self.nbm = NeuroBehaviorMat(expr=self)
        self.nbviz = NBVisualizer(self)
        self.plot_path = folder
        self.modeling_id = modeling_id
        self.cache = cache

    def meta_safe_drop(self, nb_df, inplace=False):
        subcols = list(np.setdiff1d(nb_df.columns, self.meta.columns))
        return nb_df.dropna(subset=subcols, inplace=inplace)

    @abstractmethod
    def load_animal_session(self, animal_arg, session):
        return NotImplemented

    @abstractmethod
    def encode_to_filename(self, animal, session, ftypes="processed_all"):
        return NotImplemented

    def clear_cache(self, filetype):
        find_files_recursive(self.folder, filetype, os.remove, verbose=True)

    def align_lagged_view(
        self, proj, events, laglist=None, diagnose=False, method=None, **kwargs
    ):
        """
        Given the project code specified in `proj`, look through metadata loaded through `self.info_name`
        and align each session's neural data to `events`. If specified, lag the neural data according to
        `laglist`.
        :param proj: 3 letter project code: e.g. 'UUM', 'RRM', 'BSD', 'JUV', 'FIP'
        :param events: list, list of events to be aligned to,
                            results stored in wideform columns f'{ event}_neur|...'
        :param laglist: Dict[str, Dict]
                            dictionary specifying all column features that need to be trial lagged.
                            Each column is mapped to an additional dictionary:
                             {'pre': [lags backward in time],
                              'post': ['lags forward in time'],
                              'long': [boolean for whether pivoting the df to a long-form df]}
        :param kwargs: provide additional conditions on what sessions to include through selecting certain
        variable value in metadata
        :return: all_nb_df: pd.DataFrame
                    dataframe containing neural and behavior data
        E.g.
        ```{python}
        pse_rrm = PS_Expr(data_root)
        laglist = {'action': {'pre': 2, 'post': 3},
                   'rewarded': {'pre': 2, 'post': 3},
                   f'outcome_neur': {'pre': 2, 'post': 3, 'long': True}
        laglist.update({ev: {'pre': 1, 'post': 1} for ev in pse_rrm.nbm.behavior_events})
        nb_df_rrm = pse_rrm.align_lagged_view('RRM', ['outcome'], laglist=None, cell_type='A2A')
        ```
        """
        proj_sel = self.meta["animal"].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)
        all_nb_dfs = []

        for animal, session in tqdm(
            self.meta.loc[meta_sel & proj_sel, ["animal", "session"]].values
        ):
            try:
                bmat, neuro_series = self.load_animal_session(animal, session)
            except Exception:
                logging.warning(f"Error in {animal} {session}")
                traceback.print_exc()
                bmat, neuro_series = None, None

            if bmat is None:
                logging.info(f"skipping {animal} {session}")
            else:
                try:
                    if method is None:
                        logging.info(
                            "Using lossless method for dff calculation by default"
                        )
                        met = "lossless"
                    else:
                        met = method
                    bdf, dff_df = (
                        bmat.todf(),
                        neuro_series.calculate_dff(method=met),
                    )
                    nb_df = self.nbm.align_B2N_dff_ID(bdf, dff_df, events, form="wide")
                    nb_df = self.nbm.extend_features(nb_df)
                    if diagnose:
                        sig_scores = neuro_series.diagnose_multi_channels(
                            method=met, viz=False
                        )
                        for sigch in sig_scores:
                            sigch_score_col = neuro_series.quality_metric + f"_{sigch}"
                            self.meta.loc[
                                (self.meta["animal"] == animal)
                                & (self.meta["session"] == session),
                                sigch_score_col,
                            ] = sig_scores[sigch]
                    if laglist is not None:
                        nb_df = self.nbm.lag_wide_df_ID(nb_df, laglist)
                    all_nb_dfs.append(nb_df)
                except Exception or ValueError:
                    logging.warning(
                        f"Error in calculating bdf, dff or AUC-score for {animal}, {session}, check signal"
                    )
                    traceback.print_exc()
                    bmat, neuro_series = None, None
        if not all_nb_dfs:
            return None

        all_nb_df = pd.concat(all_nb_dfs, axis=0)
        return all_nb_df.merge(self.meta, how="left", on=["animal", "session"])

    def align_lagged_view_parquet(self, parquet_file):
        """:) Tying shoe laces, loads aligned nb_df from memory instead of computing on the fly"""
        nb_df = pd.read_parquet(parquet_file)
        self.nbm.nb_cols, self.nbm.nb_lag_cols = self.nbm.parse_nb_cols(nb_df)
        nb_df.drop(
            columns=[
                c
                for c in self.meta.columns
                if (c not in ["animal", "session"]) and (c in nb_df.columns)
            ],
            inplace=True,
        )
        return nb_df.merge(self.meta, how="left", on=["animal", "session"])

    def behavior_lagged_view(self, proj, laglist=None, **kwargs):
        """
        Given the project code specified in `proj`, look through metadata loaded through `self.info_name`
        and merge all behavioral data. If specified, lag data according to `laglist`.
        :param proj: 3 letter project code: e.g. 'UUM', 'RRM', 'BSD', 'JUV', 'FIP'
        :param laglist: Dict[str, Dict]
                            dictionary specifying all column features that need to be trial lagged.
                            Each column is mapped to an additional dictionary:
                             {'pre': [lags backward in time],
                              'post': ['lags forward in time'],
                              'long': [boolean for whether pivoting the df to a long-form df]}
        :param kwargs: provide additional conditions on what sessions to include through selecting certain
        variable value in metadata
        :return: all_nb_df: pd.DataFrame
                    dataframe containing neural and behavior data
        E.g.
        ```{python}
        pse_rrm = PS_Expr(data_root)
        laglist = {'action': {'pre': 2, 'post': 3},
                   'rewarded': {'pre': 2, 'post': 3},
                   f'outcome_neur': {'pre': 2, 'post': 3, 'long': True}
        laglist.update({ev: {'pre': 1, 'post': 1} for ev in pse_rrm.nbm.behavior_events})
        nb_df_rrm = pse_rrm.align_lagged_view('RRM', ['outcome'], laglist=None, cell_type='A2A')
        ```
        """
        proj_sel = self.meta["animal"].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)
        all_bdfs = []
        errors = []

        for animal, session in self.meta.loc[
            meta_sel & proj_sel, ["animal", "session"]
        ].values:
            try:
                bmat, _ = self.load_animal_session(animal, session)
                if self.modeling_id and (bmat.modeling_id is None):
                    print(
                        f"Skipping {animal} {session} lacking model#{self.modeling_id}"
                    )
                    continue
            except Exception:
                traceback.print_exc()
                logging.warning(f"Error in {animal} {session}")
                errors.append([animal, session, "loading"])
                bmat, neuro_series = None, None

            if bmat is None:
                logging.info(f"skipping {animal} {session}")
            else:
                try:
                    bdf = bmat.todf()
                    all_bdfs.append(bdf)
                except Exception:
                    errors.append([animal, session, "bdf_error"])
                    logging.warning(
                        f"Error in computing bmat for {animal}, {session}, check session"
                    )
        if errors:
            error_df = pd.DataFrame(
                np.array(errors), columns=["animal", "session", "error"]
            )
            error_df.to_csv(os.path.join(self.folder, "error_log.csv"))
        all_bdf = pd.concat(all_bdfs, axis=0)
        final_bdf = all_bdf.merge(self.meta, how="left", on=["animal", "session"])
        if "opto_stim" in final_bdf.columns:
            final_bdf.loc[final_bdf["opto_stim"] != 1, "stimulation_on"] = None
            final_bdf.loc[final_bdf["opto_stim"] != 1, "stimulation_off"] = None
        return final_bdf

    def neural_sig_check(self, proj, method="lossless", **kwargs):
        """
        Given a project, which is the selector for what subjects are included in the analysis,
        perform photometry preprocessing and visualizes data quality for and generates a plot for
        each single session.

        proj: str, only select animals with ID of the form: '[proj]XX'
        method: str, default: 'lossless'
            specifies specific method used for preprocessing, check FP_quality_visualization function
            for more details.
        **kwargs: dataframe keyword selectors to specify sessions, e.g. cell_type='D1'
        """
        proj_sel = self.meta["animal"].str.startswith(proj)
        meta_sel = df_select_kwargs(self.meta, return_index=True, **kwargs)

        for animal, session in self.meta.loc[
            meta_sel & proj_sel, ["animal", "session"]
        ].values:
            try:
                _, neuro_series = self.load_animal_session(animal, session)
                qual_check_folder = oj(self.plot_path, "FP_quality_check")
                sig_scores = neuro_series.diagnose_multi_channels(
                    plot_path=qual_check_folder, method=method
                )
            except:
                logging.warning(f"Error in {animal} {session}")
                bmat, ps_series = None, None
