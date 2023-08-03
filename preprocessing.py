import sklearn
import pandas as pd
import numpy as np
from abc import abstractmethod
from statsmodels.tsa.tsatools import lagmat
import patsy
import h5py
import logging
import os
from behaviors import PSBehaviorMat


""" ##########################################
################# Diagnostic #################
########################################## """


def debug_behavior(expr, animal_arg, session, cache_folder):
    file_found = False
    arg_type = (
        "animal_ID" if (animal_arg in np.unique(expr.meta.animal_ID)) else "animal"
    )
    for aopt in ("animal_ID", "animal"):
        filearg = expr.meta.loc[expr.meta[arg_type] == animal_arg, aopt].values[0]
        filemap = expr.encode_to_filename(
            filearg, session, ["behaviorLOG", "FP", "FPTS"]
        )
        if filemap["behaviorLOG"] is not None:
            file_found = True
            break
    if not file_found:
        logging.warning(f"Cannot find files for {animal_arg}, {session}")
        return None, None

    hfile = h5py.File(filemap["behaviorLOG"], "r")
    animal_alias = expr.meta.loc[expr.meta[arg_type] == animal_arg, "animal"].values[0]
    cfolder = expr.folder if expr.cache else None
    bmat = PSBehaviorMat(
        animal_alias,
        session,
        hfile,
        STAGE=0,
        modeling_id=expr.modeling_id,
        cache_folder=cfolder,
    )
    bdf = bmat.eventlist.as_df()
    bdf.to_csv(os.path.join(cache_folder, f"{animal_arg}_{session}_behavior_debug.csv"))
    # bmat, _ = pse.load_animal_session(animal_arg, session)
    # bmat.eventlist.as_df().to_csv(os.path.join(cache_folder, 'debugging', f"BSD017_p166_behavior_debug.csv"))
    # bdf = bmat.todf()
    return bdf


""" ##########################################
################ Preprocessing ###############
########################################## """


class NBM_Preprocessor:
    def __init__(self, nbm, save_model=False):
        self.save_model = save_model
        self.model = None
        self.nbm = nbm
        self.RAND = self.nbm.RAND
        pass

    def transform(self):
        pass

    def get_neural_mat_wide(self, nb_df, **kwargs):
        return nb_df

    def get_neural_mat_wide_ID(self, nb_df, **kwargs):
        # assuming optimal neural event alignment window
        # dim-{i}: reduced dimension of the neural data session wise
        for kw in kwargs:
            if kw.startswith("dim-"):
                dim_method = kw.split("-")[1]

    def neural_dim_reduction(self, nb_df, event, method):
        """
        Input:
            nb_df: pd.DataFrame
                Neurobehavior data in wide form, where each row is one trial with behavior data and
                behavior-aligned neural signals
            event: str
                event name with which neural signals should be aligned and performed dim reduction
        Output:
            df_LD: pd.DataFrame
                dataframe containing dim reduced neural signals, with column names the corresponding
                reduced neural signals
        """
        # Get columns of peri-event time-stamps for neural signals
        ev_neur = self.nbm.default_ev_neur
        if (ev_neur(event) in self.nbm.nb_cols) or (
            ev_neur(event) in self.nbm.nb_lag_cols
        ):
            colnames = [c for c in nb_df.columns if ev_neur(event) in c]
            # colnames = self.nbm.nb_cols[ev_neur(event)]
        else:
            raise RuntimeError(f"Unknown event {event}")
        X = nb_df[colnames].values
        sorted_cols = np.sort(colnames)
        t_start = sorted_cols[0].split("|")[1]
        t_end = sorted_cols[-1].split("|")[1]
        event_arg = ev_neur(event)

        if method == "mean":
            df_LD = pd.DataFrame(
                {f"{event_arg}_mean({t_start},{t_end})": np.mean(X, axis=1)}
            )
        elif method == "peakridge":
            # here a greedy version of the peak ridge is computed, where it is assumed that
            # when \mu(X[i]) the summary stat is positive is the maximum, and the mininum
            # when negative
            mm = np.mean(X, axis=1)
            pr = np.empty(X.shape[0])
            pr[mm >= 0] = np.max(X[mm >= 0], axis=1)
            pr[mm < 0] = np.min(X[mm < 0], axis=1)
            df_LD = pd.DataFrame({f"{event_arg}_peakridge": pr})
        elif method == "conv_vtx":
            ab = np.abs(X)
            df_LD = pd.DataFrame(
                {
                    f"{event_arg}_conv_vtx": X[
                        np.arange(X.shape[0]), np.argmax(ab, axis=1)
                    ]
                }
            )
        elif method == 0:
            df_LD = pd.DataFrame(X, columns=colnames)
        else:
            pca = sklearn.decomposition.PCA(
                n_components=method, whiten=True, random_state=self.RAND
            )
            df_LD = pd.DataFrame(
                pca.fit_transform(X),
                columns=[f"{event_arg}_PC{j + 1}" for j in range(method)],
            )
            if self.save_model:
                self.model = pca
        return df_LD


class CV_df_Preprocessor:
    def __init__(self, engine="sklearn", **kwargs):
        self.idx = None
        self.x_cols = []
        self.y_name = None
        self.engine = engine

    def set_engine(self, mode):
        self.engine = mode

    @abstractmethod
    def fit_transform(self, data):
        raise NotImplementedError

    def index_train_test(self, train_inds, test_inds):
        x_train_inds = np.where(np.isin(self.idx, train_inds))[0]
        x_test_inds = np.where(np.isin(self.idx, test_inds))[0]
        return x_train_inds, x_test_inds

    def filter_data(self, X, y, sel):
        # filter data original data with boolean array `sel`
        idsel = np.isin(self.idx, np.where(sel)[0])[0]
        return X[idsel], y[idsel]


class PSLR_Preprocessor(CV_df_Preprocessor):
    """
    Assumes input dataframe follows sam structure described uin cogmodels_base.py

    tested: sklearn and sm.GLM LR gives similar results
    """

    def __init__(self, lag=4, engine="sklearn", **kwargs):
        super().__init__(engine, **kwargs)
        self.lag = lag

    def fit_transform(self, data):
        # version outputs transformed y data since it is selected from `data` columns
        # rdf = data[['ID', 'Session', 'Trial', 'Decision', 'Reward']].rename(columns={'Reward': 'R'}).reset_index(
        #     drop=True)
        # No need for intercept since using sklearn
        # feature engineering
        rdf = data.rename(columns={"Reward": "R"})
        rdf["C"] = 2 * rdf["Decision"] - 1
        features = ["C", "R"]
        lagfeats = list(
            np.concatenate(
                [
                    [feat + f"_{i}back" for feat in features]
                    for i in range(1, self.lag + 1)
                ]
            )
        )

        lagdf = pd.DataFrame(
            lagmat(
                rdf[features].values, maxlag=self.lag, trim="forward", original="ex"
            ),
            columns=lagfeats,
        )
        col_keys = ["C"] + [f"C_{i}back" for i in range(1, self.lag + 1)]
        lagdf = pd.concat([rdf, lagdf], axis=1)
        idx_sel = (lagdf["Trial"] > self.lag) & np.logical_and.reduce(
            [(lagdf[c] != -3) for c in col_keys]
        )
        lagdf = lagdf[idx_sel].reset_index(drop=True)
        interactions = [f"C_{i}back:R_{i}back" for i in range(1, self.lag + 1)]
        formula = "Decision ~ " + "+".join(lagfeats + interactions)
        y, X = patsy.dmatrices(formula, data=lagdf, return_type="dataframe")
        if self.engine == "sklearn":
            X.drop(columns="Intercept", inplace=True)
        self.x_cols = list(X.columns)
        self.y_name = "Decision"
        self.idx = np.where(idx_sel)[0]
        # TODO: decide if ravel completely
        return X, y


class DALR_Preprocessor(CV_df_Preprocessor):
    """
    Assumes input dataframe follows sam structure described uin cogmodels_base.py

    tested: sklearn and sm.GLM LR gives similar results
    """

    def __init__(
        self, lag=4, endog="outcome_DA_mean", y_lag=False, engine="statsmodel", **kwargs
    ):
        super().__init__(engine, **kwargs)
        self.lag = lag
        self.endog = endog
        self.y_lag = y_lag

    def fit_transform(self, data):
        # version outputs transformed y data since it is selected from `data` columns
        # rdf = data[['ID', 'Session', 'Trial', 'Decision', 'Reward']].rename(columns={'Reward': 'R'}).reset_index(
        #     drop=True)
        # No need for intercept since using sklearn
        # feature engineering
        nlag = self.lag
        rdf = (
            data[
                [
                    "animal",
                    "session",
                    "trial",
                    "action",
                    "rewarded",
                    "ego_action",
                    self.endog,
                ]
            ]
            .rename(columns={"rewarded": "R"})
            .reset_index(drop=True)
        )
        if self.endog == "outcome_DA_mean":
            self.endog = "ODA"
            rdf.rename(columns={"outcome_DA_mean": "ODA"}, inplace=True)
        elif "rpe__" in self.endog:
            old_name = self.endog
            self.endog = self.endog.replace("__", "")
            rdf.rename(columns={old_name: self.endog}, inplace=True)

        if self.y_lag:
            features = ["A", "R", self.endog]
            lagcols = ["C0", "R", self.endog]
            interactions = [f"A_{i}back:R_{i}back" for i in range(1, nlag + 1)] + [
                f"A_{i}back:{self.endog}_{i}back" for i in range(1, nlag + 1)
            ]
        else:
            features = ["A", "R"]
            lagcols = ["C0", "R"]
            interactions = [f"A_{i}back:R_{i}back" for i in range(1, nlag + 1)]

        # feature engineering
        # input: rdf with C, R columns
        rdf["R"] = rdf["R"].astype(float)
        rdf["C0"] = rdf["action"].map({"left": 0, "right": 1}).astype(float)
        rdf["contra"] = rdf["ego_action"].map({"contra": 1, "ipsi": 0})

        lagfeats = list(
            np.concatenate(
                [[feat + f"_{i}back" for feat in features] for i in range(1, nlag + 1)]
            )
        )

        lagdf = pd.DataFrame(
            lagmat(rdf[lagcols].values, maxlag=nlag, trim="forward", original="ex"),
            columns=lagfeats,
        )
        for i in range(1, nlag + 1):
            fi = f"A_{i}back"
            v = 2 * (lagdf[fi].values == rdf["C0"].values).astype(float) - 1
            v[np.isnan(lagdf[fi].values)] = np.nan
            lagdf[fi] = v

        col_keys = ["C0"] + lagfeats + ["R", "contra", self.endog]
        lagdf = pd.concat([rdf, lagdf], axis=1)
        idx_sel = (lagdf["trial"] > nlag) & np.logical_and.reduce(
            [~np.isnan(lagdf[c]) for c in col_keys]
        )
        lagdf = lagdf[idx_sel].reset_index(drop=True)

        formula = f"{self.endog} ~ " + "+".join(
            lagfeats + interactions + ["R", "contra"]
        )  # assuming no effect from past DA
        y, X = patsy.dmatrices(formula, data=lagdf, return_type="dataframe")
        if self.engine == "sklearn":
            X.drop(columns="Intercept", inplace=True)
        self.x_cols = list(X.columns)
        self.y_name = self.endog
        self.idx = np.where(idx_sel)[0]
        # TODO: decide if ravel completely
        return X, y
