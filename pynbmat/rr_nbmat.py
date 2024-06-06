import numpy as np
from neuro_series import *
from nb_viz import *
from peristimulus import *
from sklearn.model_selection import train_test_split
from pynbmat.nbmat_base import NeuroBehaviorMat, NBExperiment
import logging

logging.basicConfig(level=logging.INFO)
# sns.set_context("talk")
RAND_STATE = 230


class RR_NBMat(NeuroBehaviorMat):
    fields = [
        "tone_onset",
        "T_Entry",
        "choice",
        "outcome",
        "quit",
        "collection",
        "trial_end",
        "exit",
    ]
    # Add capacity for only behavior
    behavior_events = RRBehaviorMat.fields
    # tone_onset -> T_Entry -> choice (-> {0: quit,
    #                                      1: outcome (-> collection)}) -> trial_end

    event_features = {
        "accept": ["tone_onset", "T_entry", "choice", "trial_end", "exit"],
        "reward": ["outcome", "trial_end", "exit"],
    }

    trial_features = ["tone_prob", "restaurant", "lapIndex", "blockIndex"]

    id_vars = ["animal", "session", "roi"]
    uniq_cols = id_vars + ["trial"]

    def __init__(self, neural=True, expr=None):
        super().__init__(neural, expr)
        if not neural:
            self.id_vars = ["animal", "session"]
        self.event_time_windows = {
            "tone_onset": np.arange(-1, 1.001, 0.05),
            "T_Entry": np.arange(-1, 1.001, 0.05),
            "choice": np.arange(-1, 1.001, 0.05),
            "outcome": np.arange(-1, 1.001, 0.05),
            "" "quit": np.arange(-1, 1.001, 0.05),
            "collection": np.arange(-1, 1.001, 0.05),
            "trial_end": np.arange(-1, 1.001, 0.05),
            "exit": np.arange(-1, 1.001, 0.05),
        }

    def fit_action_value_function(self, df):
        # endogs
        exog = "accept"
        to_convert = ["restaurant"]
        X = pd.concat(
            [pd.get_dummies(df["restaurant"].astype(str)), df["tone_prob"]], axis=1
        )
        reg_df = pd.concat([X, df[to_convert]], axis=1)
        y = df[exog].values
        # Use held out dataset to evaluate score
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RAND_STATE
        )
        # clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
        clf = LogisticRegression(random_state=RAND_STATE).fit(X_train, y_train)
        cv_score = clf.score(X_test, y_test)
        # use full dataset to calculate action logits
        clf_psy = clf.fit(X, y)
        base_df = reg_df.drop_duplicates().reset_index(drop=True)
        X_base = base_df.drop(columns=to_convert)
        logits = X_base.values @ clf_psy.coef_.T + clf_psy.intercept_
        base_df["action_logit"] = logits

        def func(endog_df):
            return endog_df.merge(base_df, how="left", on=list(endog_df.columns))[
                "action_logit"
            ]

        return {
            "score": cv_score,
            "func": func,
            "name": "action_logit",
            "debug": base_df,
        }

    def add_action_value_feature(self, df, endog_map):
        endog = ["restaurant", "tone_prob"]
        df[endog_map["name"]] = endog_map["func"](df[endog])
        return df

    def add_action_value_animal_wise(self, nb_df):
        endog_map = self.fit_action_value_function(
            nb_df[nb_df["stimType"] == "nostim"].reset_index(drop=True)
        )
        reg_df = self.add_action_value_feature(nb_df, endog_map)
        return reg_df


############################################################
######################## RR EXPR ###########################
############################################################


class RR_Expr(NBExperiment):
    # TODO: for decoding, add functions to merge multiple rois
    info_name = "rr_neural_subset.csv"
    spec_name = "rr_animal_specs.csv"

    def __init__(self, folder, modeling_id=None, cache=True, **kwargs):
        super().__init__(folder, modeling_id, cache)
        self.folder = folder
        pathlist = folder.split(os.sep)[:-1] + ["plots"]
        self.plot_path = oj(os.sep, os.sep.join(pathlist))
        print(f"Changing plot_path as {self.plot_path}")
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        for kw in kwargs:
            if hasattr(self, kw):
                setattr(self, kw, kwargs[kw])
        info = pd.read_csv(os.path.join(folder, self.info_name))
        spec = pd.read_csv(os.path.join(folder, self.spec_name))
        self.meta = info.merge(spec, left_on="animal", right_on="alias", how="left")
        # # self.meta.loc[self.meta['session_num']]
        self.meta["cell_type"] = self.meta["animal_ID"].str.split("-", expand=True)[0]
        self.meta["session"] = self.meta["age"].apply(self.cvt_age_to_session)
        self.nbm = RR_NBMat(expr=self)
        self.nbviz = RR_NBViz(self)

        # # TODO: modify this later
        if ("trig_mode" not in self.meta.columns) and (
            "fp_recorded" in self.meta.columns
        ):
            self.meta.loc[self.meta["fp_recorded"] == 1, "trig_mode"] = "TRIG1"

    def cvt_age_to_session(self, age):
        DIG_LIMIT = 2  # limit the digits allowed for age representation (max 99)
        age = float(age)
        if np.allclose(age % 1, 0):
            return f"Day{int(age)}"
        else:
            digit = np.around(age % 1, DIG_LIMIT)
            agenum = int(age // 1)
            if np.allclose(digit, 0.05):
                return f"Day{agenum}_session0"
            else:
                snum = str(digit).split(".")[1]
                return f"Day{agenum}_session{snum}"

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
            filemap = self.encode_to_filename(filearg, session, ["RR_", "FP", "FPTS"])
            if filemap["RR_"] is not None:
                file_found = True
                break
        if not file_found:
            logging.warning(f"Cannot find files for {animal_arg}, {session}")
            return None, None

        animal_alias = self.meta.loc[
            self.meta[arg_type] == animal_arg, "animal"
        ].values[0]
        cfolder = self.folder if self.cache else None
        bmat = RRBehaviorMat(
            animal_alias, session, filemap["RR_"], STAGE=1, cache_folder=cfolder
        )
        fp_file = filemap["FP"]
        fp_timestamps = filemap["FPTS"]

        if (fp_file is not None) and (fp_timestamps is not None):
            session_sel = self.meta["session"] == session
            trig_mode = self.meta.loc[
                (self.meta[arg_type] == animal_arg) & session_sel, "trig_mode"
            ].values[0]
            rr_series = BonsaiRR2Hemi2Ch(
                fp_file,
                fp_timestamps,
                trig_mode,
                animal_alias,
                session,
                cache_folder=cfolder,
            )
            rr_series.merge_channels(ts_resamp_opt="interp")
            rr_series.realign_time(bmat)
            bmat.adjust_tmax(rr_series)
        else:
            rr_series = None
        return bmat, rr_series

    def encode_to_filename(self, animal, session, ftypes="all"):
        """
        :param folder: str
                folder for data storage
        :param animal: str
                animal name: e.g. A2A-15B-B_RT
        :param session: str
                session name: e.g. p151_session1_FP_RH
        :param ftype: list or str:
                list (or a single str) of typed files to return
                'RR_': behavior files
                'bin_mat': binary file
                'green': green fluorescence
                'red': red FP
                'behavior': .mat behavior file
                'FP': processed dff hdf5 file
                if ftypes=="all"
        :return:
                returns all 5 files in a dictionary; otherwise return all file types
                in a dictionary, None if not found
        """
        folder = self.folder
        # bfolder = oj(folder, 'RR_Behavior_Data')
        # fpfolder = oj(folder, 'RR_FP_Data')
        # paths = [os.path.join(bfolder, animal, session), os.path.join(bfolder, animal + '_' + session),
        #          os.path.join(bfolder, animal), bfolder,
        #          oj(fpfolder, animal, session), oj(fpfolder, animal + '_' + session),
        #          oj(fpfolder, animal), fpfolder]
        paths = [
            oj(folder, animal, session),
            oj(folder, animal + "_" + session),
            oj(folder, animal),
            folder,
        ]
        if ftypes == "all":
            ftypes = ["RR_", "FP", "FPTS"]
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


class RR_Opto(RR_Expr):
    info_name = "rr_opto_subset.csv"
    spec_name = "rr_animal_specs.csv"

    def __init__(self, folder, **kwargs):
        super().__init__(folder, **kwargs)
        self.nbm = RR_NBMat(neural=False)
