import numpy as np
from neuro_series import *
from nb_viz import *
from peristimulus import *
from sklearn.model_selection import train_test_split
from pynbmat.nbmat_base import NeuroBehaviorMat, NBExperiment
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.tsa.tsatools import lagmat
import patsy
import logging


logging.basicConfig(level=logging.INFO)
# sns.set_context("talk")
RAND_STATE = 230


class PS_Expr(NBExperiment):
    info_name = "probswitch_neural_subset.csv"
    spec_name = "probswitch_animal_specs.csv"

    def __init__(self, folder, modeling_id=None, cache=True, **kwargs):
        super().__init__(folder, modeling_id, cache)
        self.folder = folder
        pathlist = folder.split(os.sep)[:-1] + ["plots"]
        self.plot_path = oj(os.sep, *pathlist)
        print(f"Changing plot_path as {self.plot_path}")
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        for kw in kwargs:
            if hasattr(self, kw):
                setattr(self, kw, kwargs[kw])
        info = pd.read_csv(os.path.join(folder, self.info_name))
        spec = pd.read_csv(os.path.join(folder, self.spec_name))
        self.meta = info.merge(spec, left_on="animal", right_on="alias", how="left")
        self.meta["animal_ID"] = self.meta["animal_ID_y"]
        # self.meta.loc[self.meta['session_num']]
        self.meta["cell_type"] = self.meta["animal_ID"].str.split("-", expand=True)[0]
        self.meta["session"] = self.meta["age"].apply(self.cvt_age_to_session)
        self.meta["hemi"] = ""
        self.meta.loc[self.meta["plug_in"] == "R", "hemi"] = "right"
        self.meta.loc[self.meta["plug_in"] == "L", "hemi"] = "left"
        self.nbm = PS_NBMat(expr=self)

        # TODO: modify this later
        if "trig_mode" not in self.meta.columns:
            self.meta["trig_mode"] = "BSC1"

    def cvt_age_to_session(self, age):
        DIG_LIMIT = 2  # limit the digits allowed for age representation (max 99)
        age = float(age)
        if np.allclose(age % 1, 0):
            return f"p{int(age)}"
        else:
            digit = np.around(age % 1, DIG_LIMIT)
            agenum = int(age // 1)
            if np.allclose(digit, 0.05):
                return f"p{agenum}_session0"
            else:
                snum = str(digit).split(".")[1]
                return f"p{agenum}_session{snum}"

    def load_animal_session(self, animal_arg, session, options="all"):
        # load animal session according to info sheet and align to neural signal
        # left location
        # right location
        # left virus
        # right virus
        file_found = False

        arg_type = (
            "animal_ID" if (animal_arg in np.unique(self.meta.animal_ID)) else "animal"
        )
        for aopt in ("animal_ID", "animal"):
            filearg = self.meta.loc[self.meta[arg_type] == animal_arg, aopt].values[0]
            filemap = self.encode_to_filename(
                filearg, session, ["behaviorLOG", "FP", "FPTS"]
            )
            if filemap["behaviorLOG"] is not None:
                file_found = True
                break
        if not file_found:
            logging.warning(f"Cannot find files for {animal_arg}, {session}")
            return None, None

        hfile = h5py.File(filemap["behaviorLOG"], "r")
        animal_alias = self.meta.loc[
            self.meta[arg_type] == animal_arg, "animal"
        ].values[0]
        cfolder = self.folder if self.cache else None
        bmat = PSBehaviorMat(
            animal_alias,
            session,
            hfile,
            STAGE=1,
            modeling_id=self.modeling_id,
            cache_folder=cfolder,
        )
        hfile.close()
        fp_file = filemap["FP"]
        fp_timestamps = filemap["FPTS"]

        if (fp_file is not None) and (fp_timestamps is not None):
            session_sel = self.meta["session"] == session
            trig_mode = self.meta.loc[
                (self.meta[arg_type] == animal_arg) & session_sel, "trig_mode"
            ].values[0]
            ps_series = BonsaiPS1Hemi2Ch(
                fp_file,
                fp_timestamps,
                trig_mode,
                animal_alias,
                session,
                cache_folder=cfolder,
            )
            ps_series.merge_channels()
            ps_series.realign_time(bmat)
            bmat.adjust_tmax(ps_series)
        else:
            ps_series = None
        return bmat, ps_series

    def encode_to_filename(self, animal, session, ftypes="processed_all"):
        """Returns filenames requested by ftypes, or None if not found
        :param folder: str
                folder for data storage
        :param animal: str
                animal name: e.g. A2A-15B-B_RT
        :param session: str
                session name: e.g. p151_session1_FP_RH
        :param ftype: list or str:
                list (or a single str) of typed files to return
                'exper': .mat files
                'bin_mat': binary file
                'green': green fluorescence
                'red': red FP
                'behavior': .mat behavior file
                'FP': processed dff hdf5 file
                when ftypes is a single str, return the filename directly
        :return:
                returns all 5 files in a dictionary; otherwise return all file types
                in a dictionary, None if not found
        """
        folder = self.folder
        paths = [
            os.path.join(folder, animal, session),
            os.path.join(folder, animal + "_" + session),
            os.path.join(folder, animal),
            folder,
        ]
        if ftypes == "raw all":
            ftypes = ["exper", "bin_mat", "green", "red"]
        elif ftypes == "processed_all":
            ftypes = ["processed", "green", "red", "FP"]
        elif isinstance(ftypes, str):
            ftypes = [ftypes]
        results = {ft: None for ft in ftypes}
        registers = 0
        for p in paths:
            if os.path.exists(p):
                for f in os.listdir(p):
                    for ift in ftypes:
                        if ift == "FP":
                            ift_arg = "FP_"
                        else:
                            ift_arg = ift
                        if (ift_arg in f) and (animal in f) and (session in f):
                            results[ift] = os.path.join(p, f)
                            registers += 1
                            if registers == len(ftypes):
                                return results if len(results) > 1 else results[ift]
        return results if len(results) > 1 else list(results.values())[0]

    def select_region(self, nb_df, region):
        left_sel = (nb_df["hemi"] == "left") & (nb_df["left_region"] == region)
        right_sel = (nb_df["hemi"] == "right") & (nb_df["right_region"] == region)
        return nb_df[left_sel | right_sel].reset_index(drop=True)


class PS_NBMat(NeuroBehaviorMat):
    behavior_events = [
        "center_in",
        "center_out",
        "side_in",
        "outcome",
        "zeroth_side_out",
        "first_side_out",
        "last_side_out",
    ]

    event_features = {
        "action": ["side_in", "outcome", "zeroth_side_out", "first_side_out"],
        "last_side_out_side": ["last_side_out"],
    }

    trial_features = [
        "explore_complex",
        "struct_complex",
        "state",
        "rewarded",
        "trial_in_block",
        "block_num",
        "quality",
    ]

    id_vars = ["animal", "session", "roi"]
    uniq_cols = id_vars + ["trial"]

    def __init__(self, neural=True, expr=None):
        super().__init__(neural, expr)
        if not self.neural:
            self.id_vars = self.id_vars[:-1]
        self.event_time_windows = {
            "center_in": np.arange(-1, 1.001, 0.05),
            "center_out": np.arange(-1, 1.001, 0.05),
            "outcome": np.arange(-0.5, 2.001, 0.05),
            "side_in": np.arange(-0.5, 1.001, 0.05),
            "zeroth_side_out": np.arange(-0.5, 2.001, 0.05),
            "first_side_out": np.arange(-0.5, 2.001, 0.05),
            "last_side_out": np.arange(-0.5, 1.001, 0.05),
        }

    def get_perc_trial_in_block(self, nb_df):
        # TODO: shift this function to behaviorMat
        def ptib(nb_df):
            nb_df["perc_TIB"] = 0
            for ibn in np.unique(nb_df["block_num"]):
                bn_sel = nb_df["block_num"] == ibn
                mTIB = np.max(nb_df.loc[bn_sel, "trial_in_block"].values)
                nb_df.loc[bn_sel, "perc_TIB"] = (
                    nb_df.loc[bn_sel, "trial_in_block"].values / mTIB
                )
            return nb_df

        return self.apply_to_idgroups(nb_df, ptib, id_vars=["animal", "session"])

    def df2Xy(self, rdf, nlag=6):
        rdf["Decision"] = rdf["action"].map({"left": 0, "right": 1}).astype(float)
        rdf["C"] = 2 * rdf["action"] - 1
        features = ["C", "R"]
        lagfeats = list(
            np.concatenate(
                [[feat + f"_{i}back" for feat in features] for i in range(1, nlag + 1)]
            )
        )

        lagdf = pd.DataFrame(
            lagmat(rdf[features].values, maxlag=nlag, trim="forward", original="ex"),
            columns=lagfeats,
        )
        col_keys = ["C"] + [f"C_{i}back" for i in range(1, nlag + 1)]
        lagdf = pd.concat([rdf, lagdf], axis=1)
        lagdf = lagdf[
            (lagdf["Trial"] > nlag)
            & np.logical_and.reduce([(~lagdf[c].isnull()) for c in col_keys])
        ].reset_index(drop=True)
        interactions = [f"C_{i}back:R_{i}back" for i in range(1, nlag + 1)]
        formula = "Decision ~ " + "+".join(lagfeats + interactions)
        y, X = patsy.dmatrices(formula, data=lagdf, return_type="dataframe")
        id_df = lagdf[["animal", "session", "trial"]]
        return X, y, id_df

    def fit_action_value_function(self, df, nlag=6):
        # endogs

        rdf = (
            df[["animal", "session", "trial", "rewarded", "action"]]
            .rename(columns={"rewarded": "R"})
            .reset_index(drop=True)
        )
        X, y, _ = self.df2Xy(rdf, nlag=nlag)

        # Use held out dataset to evaluate score
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RAND_STATE
        )
        # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
        clf = LogisticRegressionCV(random_state=RAND_STATE).fit(X_train, y_train)
        cv_score = clf.score(X_test, y_test)
        # use full dataset to calculate action logits
        clf_psy = clf.fit(X, y)

        def func(X):
            logits = X @ clf_psy.coef_.T + clf_psy.intercept_
            return logits

        return {"score": cv_score, "func": func, "name": "action_logit"}

    def add_action_value_feature(self, df, endog_map, nlag=6):
        rdf = (
            df[["animal", "session", "trial", "rewarded", "action"]]
            .rename(columns={"rewarded": "R"})
            .reset_index(drop=True)
        )
        X, _, id_df = self.df2Xy(rdf, nlag=nlag)
        id_df[endog_map["name"]] = endog_map["func"](X)
        return df.merge(id_df, how="left", on=["animal", "session", "trial"])

    def add_action_value_animal_wise(self, nb_df, nlag=6):
        endog_map = self.fit_action_value_function(nb_df)
        reg_df = self.add_action_value_feature(nb_df, endog_map, nlag=nlag)
        return reg_df

    def get_switch_number(self, nb_df):
        # disregard miss trials
        test_df = self.lag_wide_df(nb_df, {"action": {"pre": 1}}).reset_index(drop=True)
        test_df["switch_num"] = np.nan
        test_df.loc[
            (test_df["trial"] == 1) & (~test_df["action"].isnull()), "switch_num"
        ] = 0
        test_df.loc[
            (test_df["action{t-1}"] != test_df["action"])
            & (~test_df["action"].isnull()),
            "switch_num",
        ] = 0
        assert test_df.loc[0, "trial"] == 1
        for i in range(test_df.shape[0]):
            if np.isnan(test_df.loc[i, "switch_num"]):
                if not pd.isnull(test_df.loc[i, "action"]):
                    test_df.loc[i, "switch_num"] = test_df.loc[i - 1, "switch_num"] + 1
        test_df.loc[test_df["trial"] == 1, "switch_num"] = np.nan
        return test_df.drop(columns=["action{t-1}"]).reset_index(drop=True)

    def get_reward_number(self, nb_df):
        # Treating miss as different reward outcome, disrupting reward sequence
        ru_convert = lambda x: (x - 0.5) * 2
        test_df = self.lag_wide_df(nb_df, {"rewarded": {"pre": 1}}).reset_index(
            drop=True
        )
        test_df["reward_num"] = np.nan
        first_trial_sel = (test_df["trial"] == 1) & (~test_df["rewarded"].isnull())
        test_df.loc[first_trial_sel, "reward_num"] = (
            test_df.loc[first_trial_sel, "rewarded"].astype(float) - 0.5
        ) * 2
        rew_change_sel = (test_df["rewarded{t-1}"] != test_df["rewarded"]) & (
            ~test_df["rewarded"].isnull()
        )
        test_df.loc[rew_change_sel, "reward_num"] = (
            test_df.loc[rew_change_sel, "rewarded"].astype(float) - 0.5
        ) * 2
        assert test_df.loc[0, "trial"] == 1
        for i in range(test_df.shape[0]):
            if np.isnan(test_df.loc[i, "reward_num"]) and (
                not pd.isnull(test_df.loc[i, "rewarded"])
            ):
                test_df.loc[i, "reward_num"] = test_df.loc[
                    i - 1, "reward_num"
                ] + ru_convert(float(test_df.loc[i, "rewarded"]))
        return test_df.drop(columns=["rewarded{t-1}"]).reset_index(drop=True)

    def get_reward_switch_number(self, nb_df):
        ru_convert = lambda x: (x - 0.5) * 2
        test_df = self.lag_wide_df(
            nb_df, {"action": {"pre": 1}, "rewarded": {"pre": 1}}
        ).reset_index(drop=True)
        test_df[["switch_num", "reward_num"]] = np.nan
        test_df.loc[
            (test_df["trial"] == 1) & (~test_df["action"].isnull()), "switch_num"
        ] = 0
        test_df.loc[
            (test_df["action{t-1}"] != test_df["action"])
            & (~test_df["action"].isnull()),
            "switch_num",
        ] = 0
        first_trial_sel = (test_df["trial"] == 1) & (~test_df["rewarded"].isnull())
        test_df.loc[first_trial_sel, "reward_num"] = (
            test_df.loc[first_trial_sel, "rewarded"].astype(float) - 0.5
        ) * 2
        rew_change_sel = (test_df["rewarded{t-1}"] != test_df["rewarded"]) & (
            ~test_df["rewarded"].isnull()
        )
        test_df.loc[rew_change_sel, "reward_num"] = (
            test_df.loc[rew_change_sel, "rewarded"].astype(float) - 0.5
        ) * 2
        assert test_df.loc[0, "trial"] == 1
        for i in range(test_df.shape[0]):
            if np.isnan(test_df.loc[i, "switch_num"]) and (
                not pd.isnull(test_df.loc[i, "action"])
            ):
                test_df.loc[i, "switch_num"] = test_df.loc[i - 1, "switch_num"] + 1
            if np.isnan(test_df.loc[i, "reward_num"]) and (
                not pd.isnull(test_df.loc[i, "rewarded"])
            ):
                test_df.loc[i, "reward_num"] = test_df.loc[
                    i - 1, "reward_num"
                ] + ru_convert(float(test_df.loc[i, "rewarded"]))
        test_df.loc[test_df["trial"] == 1, "switch_num"] = np.nan
        test_df["switch"] = np.nan
        test_df.loc[test_df["switch_num"] == 0, "switch"] = True
        test_df.loc[test_df["switch_num"] > 0, "switch"] = False
        return test_df.drop(columns=["action{t-1}", "rewarded{t-1}"]).reset_index(
            drop=True
        )

    def get_OHSH(self, nb_df):
        test_df = self.lag_wide_df(
            nb_df, {"switch": {"pre": 2}, "rewarded": {"pre": 2}}
        ).reset_index(drop=True)
        key_cols = ["rewarded{t-1}", "rewarded{t-2}", "switch{t-1}", "switch{t-2}"]
        nonancols = ~np.any(pd.isnull(test_df[key_cols[:2]].values), axis=1)
        temp_df = test_df.loc[nonancols, key_cols[:2]].reset_index(drop=True)
        temp_df[key_cols[:2]] = temp_df[key_cols[:2]].astype(bool)

        temp_df.loc[temp_df["rewarded{t-2}"] & temp_df["rewarded{t-1}"], "OH"] = "RR"
        temp_df.loc[(~temp_df["rewarded{t-2}"]) & temp_df["rewarded{t-1}"], "OH"] = "UR"
        temp_df.loc[(~temp_df["rewarded{t-2}"]) & (~temp_df["rewarded{t-1}"]), "OH"] = (
            "UU"
        )
        temp_df.loc[temp_df["rewarded{t-2}"] & (~temp_df["rewarded{t-1}"]), "OH"] = "RU"
        test_df.loc[nonancols, "OH"] = temp_df["OH"].values
        # nonan cols for switch history
        nonancols = ~np.any(pd.isnull(test_df[key_cols[-2:]].values), axis=1)
        temp_df = test_df.loc[nonancols, key_cols[-2:]].reset_index(drop=True)
        temp_df[key_cols[-2:]] = temp_df[key_cols[-2:]].astype(bool)

        temp_df.loc[temp_df["switch{t-2}"] & temp_df["switch{t-1}"], "SH"] = "YY"
        temp_df.loc[(~temp_df["switch{t-2}"]) & temp_df["switch{t-1}"], "SH"] = "NY"
        temp_df.loc[(~temp_df["switch{t-2}"]) & (~temp_df["switch{t-1}"]), "SH"] = "NN"
        temp_df.loc[temp_df["switch{t-2}"] & (~temp_df["switch{t-1}"]), "SH"] = "YN"
        test_df.loc[nonancols, "SH"] = temp_df["SH"].values
        return test_df

    def extend_features(self, nb_df, *args, **kwargs):
        nb_df = self.get_perc_trial_in_block(nb_df)
        nb_df["pTIB_Q"] = pd.cut(nb_df["perc_TIB"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        return nb_df
