# System
import os.path
from abc import abstractmethod

# Data
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
from scipy import interpolate

# Plotting
import matplotlib.pyplot as plt

# Utils
from utils import *
from behavior_base import PSENode, EventNode
from packages.RR_bmat.eventcodedict import eventcodedict_full as RR_codemap
from packages.RR_bmat.mainAnalysis import *
from packages.RR_bmat.eventcodedict import *
from packages.RR_bmat.clean_bonsai_output import *


#######################################################
################### Data Structure ####################
#######################################################


class BehaviorMat:
    code_map = {}
    fields = []  # maybe divide to event, ev_features, trial_features
    time_unit = None
    eventlist = None

    def __init__(self, animal, session, cache_folder=None):
        self.animal = animal
        self.session = session
        self.time_aligner = lambda s: s  # provides method to align timestamps
        self.tmax = 0
        self.cache_folder = (
            os.path.join(cache_folder, animal, session)
            if (cache_folder is not None)
            else None
        )

    @abstractmethod
    def todf(self):
        return NotImplemented

    def align_ts2behavior(self, timestamps):
        return self.time_aligner(timestamps)

    def adjust_tmax(self, neuro_series):
        return max(self.tmax, np.max(neuro_series.neural_df["time"]))


class RRBehaviorMat(BehaviorMat):
    """
    STAGE: 0, raw behavior log
    1, cleaned partial behavior log
    2, trial structure with pseudo trials
    3, trial structure without pseudo trials
    """

    code_map = RR_codemap
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
    time_unit = "s"

    def __init__(self, animal, session, logfile, STAGE=1, cache_folder=None):
        super().__init__(animal, session, cache_folder=cache_folder)
        names = ["timestamp", "eventcode"]
        strip = lambda t: t.replace(" ", "") if isinstance(t, str) else t
        bonsai_output = pd.read_csv(logfile, sep=" ", index_col=False, names=names)[
            names
        ]
        bonsai_output["timestamp"] = bonsai_output["timestamp"].map(strip).astype(float)
        self.time_aligner = lambda ts: (ts - bonsai_output.iloc[0, 0]) / 1000
        self.tmax = self.time_aligner(np.max(bonsai_output["timestamp"].values))
        self.events = preprocessing(logfile, eventcodedict_full)
        self.eventlist = self.initialize(logfile, stage=STAGE)

    def initialize(self, logfile, stage=1):
        if stage == 0:
            # Save raw bonsai output with event description --> raw behavior LOG human readable
            return write_bonsaiEvent_dll(self.events)
        assert stage == 1, f"Unknown stage {stage}"
        # Save selected bonsai events --> cleaned behavior LOG, dropping nonsense
        events_partial = detect_keyword_in_event(
            preprocessing(logfile, eventcodedict_partial)
        )
        events_list_partial = clean_and_organize(events_partial)
        return write_bonsaiEvent_dll(events_list_partial)

    def todf(self, valid=True, comment=False, overwrite=False):
        # Don't use todf if initialized with STAGE 0
        # trial structure containing pseudotrials
        cache_file = None
        if self.cache_folder is not None:
            cache_file = os.path.join(
                self.cache_folder, f"{self.animal}_{self.session}_bdf.pq"
            )
            if os.path.exists(cache_file) and (not overwrite):
                return pd.read_parquet(cache_file)
        trials = trial_writer(self.eventlist)
        trial_info_filler(trials)
        trial_merger(trials)
        write_lap_block(trials)
        resort_trial_DLL(trials)
        add_stimulation_events(trials, self.events)
        trials_df = write_trial_to_df(trials)
        if valid:
            result_df = save_valid_trial(trials_df).reset_index(drop=True)
            # new_df = trials_df[trials_df.trial_end.notnull()]
            # result_df = new_df.sort_values(by='tone_onset').reset_index(drop=True)
        else:
            result_df = trials_df.reset_index(drop=True)
        if not comment:
            result_df.drop(columns="comment", inplace=True)
        for ev in self.fields:
            result_df[ev] = result_df[ev].astype(float)
        result_df["tone_prob"] = result_df["tone_prob"].astype(float)
        result_df.rename(columns={"tone_prob": "offer_prob"}, inplace=True)
        result_df["quit_time"] = result_df["quit"] - result_df["choice"]
        # result_df["offer_wait"] = result_df["tone_prob"].map(
        #     {0.0: 7, 20.0: 5, 80.0: 3, 100.0: 1}
        # )
        old_cols = list(result_df.columns)
        result_df["animal"] = self.animal
        result_df["session"] = self.session
        result_df["trial"] = np.arange(1, result_df.shape[0] + 1)
        result_df = result_df[["animal", "session", "trial"] + old_cols]
        result_df["tmax"] = self.tmax
        if cache_file is not None:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
            result_df.to_parquet(cache_file)
        return result_df

    def eventlist_to_df(self):
        # non-prefered method but use it for convenience
        return write_dll_to_df(self.eventlist)


class PSBehaviorMat(BehaviorMat):
    # Behavior Mat for Probswitch
    # Figure out how to make it general
    code_map = {
        1: ("center_in", "center_in"),
        11: ("center_in", "initiate"),
        2: ("center_out", "center_out"),
        3: ("side_in", "left"),
        4: ("side_out", "left"),
        44: ("side_out", "left"),
        5: ("side_in", "right"),
        6: ("side_out", "right"),
        66: ("side_out", "right"),
        71.1: ("outcome", "correct_unrewarded"),
        71.2: ("outcome", "correct_rewarded"),
        72: ("outcome", "incorrect_unrewarded"),
        73: ("outcome", "missed"),  # saliency questionable
        74: ("outcome", "abort"),
    }  # saliency questionable

    # divide things into events, event_features, trial_features
    fields = [
        "center_in",
        "center_out",
        "side_in",
        "outcome",
        "zeroth_side_out",
        "first_side_out",
        "last_side_out",
    ]  # 'ITI'

    time_unit = "s"

    # event_features = 'reward', 'action',
    # trial_features = 'quality', 'struct_complex', 'explore_complex', 'BLKNo', 'CPort'
    # Always use efficient coding
    def __init__(
        self,
        animal,
        session,
        hfile,
        tau=np.inf,
        STAGE=1,
        modeling_id=None,
        cache_folder=None,
    ):
        super().__init__(animal, session, cache_folder=cache_folder)
        self.tau = tau
        if isinstance(hfile, str):
            print("For pipeline loaded hdf5 is recommended for performance")
            hfile = h5py.File(hfile, "r")
            hfname = hfile
        else:
            hfname = hfile.filename
        self.animal = animal
        self.session = session
        self.choice_sides = None
        self.trialN = len(hfile["out/outcome"])
        self.modeling_id = modeling_id
        self.folder = os.path.join(
            os.sep, *hfname.split(os.path.sep)[:-1]
        )  # DEFAULT absolute path
        animal, session, modeling_id = self.animal, self.session, self.modeling_id
        model_file = os.path.join(
            self.folder, f"{animal}_{session}_modeling_{modeling_id}.hdf5"
        )
        if not os.path.exists(model_file):
            self.modeling_id = None
        self.eventlist = self.initialize_PSEnode(hfile, stage=STAGE)
        self.correct_port = self.get_correct_port_side(hfile)
        if "digital_LV_time" in hfile["out"]:
            self.time_aligner = interpolate.interp1d(
                np.array(hfile["out/digital_LV_time"]).ravel(),
                np.array(hfile["out/exper_LV_time"]).ravel(),
                fill_value="extrapolate",
            )

        switch_inds = np.full(self.trialN, False)
        switch_inds[1:] = self.correct_port[1:] != self.correct_port[:-1]
        t_in_block = np.full(self.trialN, 0)
        block_number = np.full(self.trialN, 1)
        for i in range(1, len(switch_inds)):
            if not switch_inds[i]:
                t_in_block[i] = t_in_block[i - 1] + 1
                block_number[i] = block_number[i - 1]
            else:
                block_number[i] = block_number[i - 1] + 1
        self.block_num = block_number
        self.t_in_block = t_in_block
        self.prebswitch_num = self.get_prebswitch_num(switch_inds)

    def __str__(self):
        return f"BehaviorMat({self.animal}_{self.session}, tau={self.tau})"

    def get_correct_port_side(self, hfile):
        # right: 1, left: 2
        portside = np.array(hfile["out/cue_port_side"])[:, 0]
        res = np.full(len(portside), "right")
        res[portside == 2] = "left"
        return res

    def get_prebswitch_num(self, switch_inds):
        prebswitch_num = np.full(len(switch_inds), np.nan)
        prebswitch_num[switch_inds] = 0
        switch_on = False
        for i in range(1, len(switch_inds)):
            j = len(switch_inds) - 1 - i
            if prebswitch_num[j] != 0:
                if ~np.isnan(prebswitch_num[j + 1]):
                    prebswitch_num[j] = prebswitch_num[j + 1] - 1
        return prebswitch_num

    def get_modeling_pdf(self):
        # model_file = encode_to_filename(folder, animal, session, ['modeling'])
        animal, session, modeling_id = self.animal, self.session, self.modeling_id
        model_file = os.path.join(
            self.folder, f"{animal}_{session}_modeling_{modeling_id}.hdf5"
        )
        # BRL_latents_rpe = model_file['BRL']['latent']
        hfile = h5py.File(model_file, "r")
        all_data = []
        all_data_names = []
        for mdl in hfile:
            dataset = hfile[mdl]
            for latent in dataset:
                data = np.array(dataset[latent])
                if mdl == "RAW":
                    data = data.T
                    data_name = ["RAW_" + latent]
                elif len(data.shape) == 3:
                    orig_shape = data.shape[:2]
                    data = np.reshape(data, (-1, data.shape[-1]), order="C")
                    data_name = [
                        f"{mdl}_{latent}{i}{j}"
                        for i in range(1, orig_shape[0] + 1)
                        for j in range(1, orig_shape[1] + 1)
                    ]
                elif data.shape[0] > 1:
                    data_name = [
                        f"{mdl}_{latent}{i}" for i in range(1, data.shape[0] + 1)
                    ]
                else:
                    data_name = [f"{mdl}_{latent}"]
                data = data.T
                all_data.append(data)
                all_data_names.append(data_name)
        hfile.close()
        modeling_pdf = pd.DataFrame(
            np.hstack(all_data), columns=np.concatenate(all_data_names)
        )
        return modeling_pdf

    def initialize_PSEnode(self, hfile, stage=1):
        code_map = self.code_map
        eventlist = PSENode(None, None, None, None)
        trial_event_mat = np.array(hfile["out/trial_event_mat"])
        self.tmax = np.max(trial_event_mat[:, 1])
        trialN = len(hfile["out/outcome"])
        exp_complexity = np.full(
            trialN, True, dtype=bool
        )  # default true detect back to back
        struct_complexity = np.full(
            trialN, False, dtype=bool
        )  # default false detect double centers
        prev_node = None
        for i in range(len(trial_event_mat)):
            eventcode, etime, trial = trial_event_mat[i, :]
            if stage == 0:
                event_wt = code_map[eventcode][0] + "|" + code_map[eventcode][1]
            else:
                event_wt = code_map[eventcode][0]
            # check duplicate timestamps
            if prev_node is not None:
                if prev_node.etime == etime:
                    if eventcode == prev_node.ecode:
                        continue
                    elif eventcode < 70:
                        logging.warning(
                            f"Warning! Duplicate timestamps({prev_node.ecode}, {eventcode}) in {self.animal}, {self.session}"
                        )
                    elif eventcode != 72:
                        logging.warning(
                            f"Special Event Duplicate: {self.animal}, {self.session}, {code_map[eventcode]}"
                        )
                elif eventcode == 72:
                    logging.warning(
                        f"Unexpected non-duplicate for {trial}, {code_map[eventcode]}, {self.animal}, "
                        f"{self.session}"
                    )
            cnode = PSENode(event_wt, etime, trial, eventcode)
            eventlist.append(cnode)
            prev_node = cnode
        if stage == 1:
            # skip the actual temporal merge for this stage
            runloop = True
            while runloop:
                runloop = False
                for node in eventlist:
                    # first see initiate
                    if node.ecode == 11:
                        node.saliency = code_map[node.ecode][1]
                    elif node.ecode > 70:
                        node.saliency = code_map[node.ecode][1]
                        # look backward in time and label side_in and center out
                        curr_node = node.prev
                        if node.ecode != 73:
                            # swap curr_node and prev_node label (negative duration between outcome and zero_sideout)
                            # if sideout followed by outcome
                            if curr_node.event == "side_out":
                                logging.warning(
                                    f"swapping {str(node.prev)} and {str(node)}, {self.animal}, {self.session}"
                                )
                                curr_node.trial += 0.5
                                eventlist.swap_nodes(node.prev, node)
                                runloop = True  # rerun the loop
                                break
                            assert (
                                curr_node.event == "side_in"
                            ), f"not a side_in node {str(node.prev)} preceding {str(node)}, {self.animal}, {self.session}"
                            curr_node.saliency = code_map[curr_node.ecode][1]
                        while curr_node.event != "center_out":
                            curr_node = curr_node.prev
                            if curr_node.ecode == 11:
                                raise RuntimeError(
                                    f"Center in not followed by center_out? {curr_node}"
                                )
                        curr_node.saliency = code_map[curr_node.ecode][1]
                        # look forward in time and label side_outs
                        curr_node = node.next
                        if node.ecode == 73:
                            logging.info(
                                f"skipping side_out events at miss trial {node.trial_index() + 1}"
                            )
                            continue
                            # FT: current version ignores the side out events after miss trials
                            # # for missed trial, see if the animal goes straight to the next trial
                            # while (curr_node.event != 'side_out') and (not curr_node.is_sentinel):
                            #     curr_node = curr_node.next
                            # if curr_node.is_sentinel:
                            #     assert node.trial == trialN, f'should have reached end of experiment? {str(node)}'
                            #     continue
                            # elif curr_node.trial_index() != node.trial_index():
                            #     print(f'animal straight went to the next trial from missed trial {str(node)}')
                            #     continue
                        if curr_node.is_sentinel:
                            logging.warning(
                                f"side_out after the last trial outcome is omitted at trial {node.trial_index() + 1}. {self.animal}, {self.session}"
                            )
                            continue
                        assert (
                            curr_node.event == "side_out"
                        ), f"side_out not following outcome? {str(curr_node), str(curr_node.prev)}, {self.animal}, {self.session}"
                        curr_node.saliency = code_map[curr_node.ecode][1] + "_zeroth"
                        start_node = curr_node
                        side_ecoder = lambda node: (
                            (node.ecode % 10)
                            if (node.event in ["side_in", "side_out"])
                            else node.ecode
                        )
                        # forward loop
                        while side_ecoder(curr_node) in [
                            side_ecoder(start_node),
                            side_ecoder(start_node) - 1,
                        ]:
                            curr_node = curr_node.next
                        if curr_node.prev.saliency is None:
                            curr_node.prev.saliency = code_map[curr_node.prev.ecode][1]
                        curr_node.prev.saliency += "_first"  # TODO: add TAU function to make things more rigorous
                        while (not curr_node.is_sentinel) and (curr_node.ecode != 11):
                            curr_node = curr_node.next
                        # backward loop
                        end_node = curr_node.prev
                        curr_node = end_node
                        while curr_node.event != "side_out":
                            if curr_node.event == "outcome":
                                logging.warning(
                                    f"non-missed non-terminal outcome nodes not followed by side_out at trial {node.trial_index() + 1}"
                                )
                                continue
                            curr_node = curr_node.prev

                        # now curr_node is the last side_out
                        # TODO: bug with last! figure out how this works
                        if curr_node.saliency is None:
                            curr_node.saliency = code_map[curr_node.ecode][1]
                        curr_node.saliency += "_last"
        return eventlist

    def todf(self):
        cache_file = None
        if self.cache_folder is not None:
            cache_file = os.path.join(
                self.cache_folder, f"{self.animal}_{self.session}_bdf.pq"
            )
            if os.path.exists(cache_file):
                return pd.read_parquet(cache_file)
        # careful with the trials if their last outcome is the end of the exper file.
        elist = self.eventlist
        # reward and action

        result_df = pd.DataFrame(
            np.full((self.trialN, 8), np.nan), columns=["trial"] + self.fields
        )
        result_df["animal"] = self.animal
        result_df["session"] = self.session
        result_df = result_df[["animal", "session", "trial"] + self.fields]
        result_df["trial"] = np.arange(1, self.trialN + 1)

        result_df["action"] = pd.Categorical(
            [""] * self.trialN, ["left", "right"], ordered=False
        )
        result_df["rewarded"] = np.zeros(self.trialN, dtype=bool)
        result_df["trial_in_block"] = self.t_in_block
        result_df["prebswitch_num"] = self.prebswitch_num
        result_df["block_num"] = self.block_num
        result_df["state"] = pd.Categorical(self.correct_port, ordered=False)
        result_df["quality"] = pd.Categorical(
            ["normal"] * self.trialN, ["missed", "abort", "normal"], ordered=False
        )
        result_df["last_side_out_side"] = pd.Categorical(
            [""] * self.trialN, ["left", "right"], ordered=False
        )
        for node in elist:
            if node.saliency:
                if node.event in ["center_in", "center_out"]:
                    result_df.loc[node.trial_index(), node.event] = node.etime
                elif node.event == "side_in":
                    # TODO: triple check why this is needed
                    result_df.loc[node.trial_index(), node.event] = node.etime
                    if "_" in node.saliency:
                        result_df.loc[node.trial_index(), "action"] = (
                            node.saliency.split("_")[0]
                        )
                    else:
                        result_df.loc[node.trial_index(), "action"] = node.saliency
                elif node.event == "outcome":
                    result_df.loc[node.trial_index(), node.event] = node.etime
                    result_df.loc[node.trial_index(), "rewarded"] = (
                        "_rewarded" in node.saliency
                    )
                    if node.saliency in ["missed", "abort"]:
                        result_df.loc[node.trial_index(), "quality"] = node.saliency
                        result_df.loc[node.trial_index(), "rewarded"] = np.nan
                elif node.event == "side_out":
                    if node.trial % 1 == 0.5:
                        trial_ind = int(np.floor(node.trial)) - 1
                    else:
                        print("why does this happen")
                        trial_ind = node.trial_index()
                    assert trial_ind >= 0, f"salient side_out at {str(node)}"
                    sals = node.saliency.split("_")

                    for sal in sals[1:]:
                        result_df.loc[trial_ind, sal + "_side_out"] = node.etime
                        if sal == "last":
                            result_df.loc[trial_ind, "last_side_out_side"] = sals[0]

        # STRUCT/EXP_COMPLEXITY computed on demand
        struct_complexity = np.full(
            self.trialN, False, dtype=bool
        )  # default false detect double centers
        sc_inds = np.unique(
            [
                node.trial_index()
                for node in elist
                if (node.trial % 1 == 0.5) and (node.ecode == 1)
            ]
        )
        struct_complexity[sc_inds] = True

        result_df["struct_complex"] = struct_complexity
        result_df["explore_complex"] = (
            result_df["first_side_out"].values != result_df["last_side_out"].values
        )
        if self.modeling_id:
            mdf = self.get_modeling_pdf()
            action_sel = ~result_df.action.isnull()
            assert np.sum(action_sel) == len(
                mdf
            ), f"modeling dimension mismatch for {self.animal}, {self.session}"
            result_df.loc[action_sel, list(mdf.columns)] = mdf.values
        result_df["tmax"] = self.tmax
        if cache_file is not None:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
            result_df.to_parquet(cache_file)
        return result_df


class BehaviorMatChris(BehaviorMat):
    # Figure out how to make it general

    code_map = {
        1: ("center_in", "center_in"),
        11: ("center_in", "initiate"),
        2: ("center_out", "center_out"),
        3: ("side_in", "left"),
        4: ("side_out", "left"),
        44: ("side_out", "left"),
        5: ("side_in", "right"),
        6: ("side_out", "right"),
        66: ("side_out", "right"),
        71.1: ("outcome", "correct_unrewarded"),
        71.2: ("outcome", "correct_rewarded"),
        72: ("outcome", "incorrect_unrewarded"),
        73: ("outcome", "missed"),  # saliency questionable
        74: ("outcome", "abort"),
    }  # saliency questionable

    # Always use efficient coding
    def __init__(self, animal, session, hfile, tau=np.inf):
        self.tau = tau
        self.animal = animal
        self.session = session
        if isinstance(hfile, str):
            print("For pipeline loaded hdf5 is recommended for performance")
            hfile = h5py.File(hfile, "r")
        self.choice_sides = None
        self.exp_complexity = (
            None  # Whether the ITI is complex (first round only analysis simple trials)
        )
        self.struct_complexity = None
        self.trialN = 0
        self.hemisphere, self.region = None, None
        self.event_list = EventNode(None, None, None, None)
        self.initialize(hfile)
        super().__init__(animal, session, hfile, tau)

    def __str__(self):
        return f"BehaviorMat({self.animal}_{self.session}, tau={self.tau})"

    def initialize(self, hfile):
        # TODO: reimplement for chris version
        self.hemisphere = (
            "right" if np.array(hfile["out/notes/hemisphere"]).item() else "left"
        )
        self.region = "NAc" if np.array(hfile["out/notes/region"]).item() else "DMS"
        trialN = len(hfile["out/value/outcome"])
        self.trialN = trialN
        self.choice_sides = np.full(trialN, "", dtype="<U6")
        self.exp_complexity = np.full(
            trialN, True, dtype=bool
        )  # default true detect back to back
        self.struct_complexity = np.full(
            trialN, False, dtype=bool
        )  # default false detect double centers
        self.exp_complexity[0] = False
        #         dup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
        #                'missed': 0, 'abort': 0}
        #         ndup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
        #                'missed': 0, 'abort': 0}
        #         self.struct_complexity[0] = False
        trial_event_mat = np.array(hfile["out/value/trial_event_mat"])

        # Parsing LinkedList
        prev_node = None
        # TODO: Careful of the 0.5 trial events
        for i in range(trial_event_mat.shape[0]):
            eventcode, etime, trial = trial_event_mat[i, :]
            if eventcode == 44 or eventcode == 66:
                eventcode = eventcode // 10
            ctrial = int(np.ceil(trial)) - 1
            event, opt = BehaviorMat.code_map[eventcode]
            makenew = True

            if prev_node is not None:
                if eventcode > 70:
                    lat = prev_node.MLAT if eventcode < 73 else ""
                    self.choice_sides[ctrial] = lat
                    if prev_node.event == "side_in":
                        prev_node.saliency = "choice"
                if prev_node.etime == etime:
                    if eventcode == prev_node.ecode:
                        makenew = False
                    elif eventcode < 70:
                        print(
                            f"Warning! Duplicate timestamps({prev_node.ecode}, {eventcode}) in {str(self)}"
                        )
                    elif eventcode != 72:
                        print(
                            f"Special Event Duplicate: {self.animal}, {self.session}, ",
                            event,
                            opt,
                        )
                elif eventcode == 72:
                    print(
                        f"Unexpected non-duplicate for {trial}, {opt}, {self.animal}, {self.session}"
                    )
            else:
                assert eventcode < 70, "outcome cannot be the first node"

            if makenew:
                # potentially fill out all properties here; then make merge an inheriting process
                evnode = self.event_list.append(event, etime, trial, eventcode)
                # Filling MLAT for side ports, Saliency for outcome and initiate
                if event == "outcome":
                    assert self.choice_sides[ctrial] == prev_node.MLAT
                    evnode.MLAT = prev_node.MLAT
                if eventcode > 6:
                    evnode.saliency = opt
                elif eventcode > 2:
                    evnode.MLAT = opt
                prev_node = evnode

        # temporal adjacency merge
        assert not self.event_list.is_empty()
        curr_node = self.event_list.next
        while not curr_node.sentinel:
            if "_out" in curr_node.event:
                # COULD do an inner loop to make it look more straightforward
                next_node = curr_node.next
                prev_check = curr_node.prev
                if next_node.sentinel:
                    print(f"Weird early termination with port_out?! {str(curr_node)}")
                # TODO: sanity check: choice side_in does not have any mergeable port before them.
                if (next_node.ecode == curr_node.ecode - 1) and (
                    next_node.etime - curr_node.etime < self.tau
                ):
                    merge_node = next_node.next
                    if merge_node.sentinel:
                        print(
                            f"Weird early termination with port_in?! {str(next_node)}"
                        )
                    assert (
                        merge_node.ecode == curr_node.ecode
                    ), f"side in results in {str(merge_node)}"
                    merge_node.merged = True
                    self.event_list.remove_node(curr_node)
                    self.event_list.remove_node(next_node)
                    assert (
                        prev_check.next is merge_node and merge_node.prev is prev_check
                    ), "Data Structure BUG"
                    curr_node = prev_check  # jump back to previous node

            # Mark features so far saliency: only choice/outcome/initiate, MLAT: outcome/side_port
            if (
                not curr_node.next.merged
            ):  # only trigger at "boundary events" (no new merge happened)
                # Make sure this is not a revisit due to merge
                prev_node = curr_node.prev
                next_node = curr_node.next
                if curr_node.event == "center_in":
                    # just need MLAT
                    if prev_node.event == "side_out":
                        curr_node.MLAT = prev_node.MLAT
                    # update structural complexity
                    if curr_node.saliency == "initiate":
                        breakflag = False
                        cursor = curr_node.prev
                        while (not cursor.sentinel) and (cursor.event != "outcome"):
                            if cursor.event == "center_in":
                                self.struct_complexity[curr_node.trial_index()] = True
                                breakflag = True
                                break
                            cursor = cursor.prev
                        if not breakflag and cursor.MLAT:
                            assert cursor.sentinel or (
                                cursor.next.event == "side_out"
                            ), f"weird {cursor}, {cursor.next}"
                elif curr_node.event == "center_out":
                    if next_node.event == "side_in":
                        curr_node.MLAT = next_node.MLAT
                    if next_node.saliency == "choice":
                        # assume "execution" is at center_out, recognizing that well trained animal might
                        # already have executed a program from side_out (denote side port using first/last)
                        curr_node.saliency = "execution"
                elif curr_node.event == "side_out":
                    sals = []
                    # TODO: with different TAU we might not want the first side out as salient event
                    if prev_node.event == "outcome":
                        sals.append("first")
                    if next_node.event == "center_in":
                        safe_last = True
                        cursor = next_node
                        while cursor.saliency != "initiate":
                            if cursor.sentinel:
                                print(f"Weird early termination?! {str(cursor.prev)}")
                            if cursor.event == "side_in":
                                safe_last = False
                                break
                            cursor = cursor.next
                        if safe_last:
                            sals.append("last")
                    curr_node.saliency = "_".join(sals)
                    if len(sals) == 2:
                        self.exp_complexity[int(curr_node.trial)] = False
            curr_node = curr_node.next

    def todf(self):
        elist = self.event_list
        if elist.is_empty():
            return None
        fields = [
            "trial",
            "center_in",
            "center_out",
            "side_in",
            "outcome",
            "side_out",
            "ITI",
            "A",
            "R",
            "BLKNo",
            "CPort",
        ]

    def get_event_nodes(self, event, simple=True, saliency=True):
        # TODO: replace maybe with a DataFrame implementation
        """Takes in event and returns the requested event nodes
        There are in total 3 scenarios:
        1. saliency = True, simple = True (default):
            Returns only salient event in simple trial corresponding to classic 2ABR task structure:
            outcome{t-1} -> side_out{t} (same side, first_last) -> center_in{t} (initiate)
            -> center_out{t} (execute) -> side_in{t} (choice) -> outcome{t}
            Discards trials with multiple side expoloration during ITI and non-salient events that do not
            belong to a typical task structure
        2. saliency = True, simple = False (superset of prev):
            Returns salient events in trials; Note: in outcome and choice, due to presence of miss
            trial and abort trials, the amount of entry might be less than other types
            To obtain just non-simple salient events use the following:
            ```
            event_times_sal_simp, trials_sal_simp = bmat.get_event_times('side_out')
            event_times_sal, trials_sal = bmat.get_event_times('side_out', simple=False)
            event_nodes_sal = bmat.get_event_nodes('side_out', simple=False)
            simp_sel = np.isin(event_times_sal, event_times_sal_simp)
            simp_where = np.where(simp_sel)[0]
            non_simp_etimes, non_simp_trials = event_times_sal[~simp_sel], trials_sal[~simp_sel]
            non_simp_enodes = [event_nodes_sal[en] for en in simp_where]
            # And use selectors on np.array of event nodes
            ```
        3. saliency = False, simple = False (superset of prev):
            Returns all events regardless of saliency or simplicity
            To obtain just non salient events in all trials, use similar code to above
        :param event:
        :param simple:
        :param saliency:
        :return:
        """
        curr = self.event_list.next
        event_nodes = []
        sals = None
        if simple:
            assert saliency, "no use to ensure simplicity with non-salient events"
        if saliency and "side_out" in event:
            event, sals = event.split("__")
            if sals == "":
                sals = ["first_last"]
                assert (
                    simple
                ), "no specific saliency specified for side_out, assume simple trial"
            else:
                sals = [sals, "first_last"]
        else:
            salmap = {
                "center_in": "initiate",
                "center_out": "execution",
                "side_in": "choice",
                "outcome": [
                    "correct_unrewarded",
                    "correct_rewarded",
                    "incorrect_unrewarded",
                ],
            }
            sals = salmap[event]

        while not curr.sentinel:
            if curr.event == event:
                complex_ck = True  # flag for passing the complexity check (irrelevant if simple==False)
                cti = curr.trial_index()
                if (
                    simple
                    and event in ["center_in", "side_out"]
                    and (self.exp_complexity[cti] or self.struct_complexity[cti])
                ):
                    complex_ck = False

                if (
                    (not saliency) or (curr.saliency != "" and curr.saliency in sals)
                ) and complex_ck:
                    event_nodes.append(curr)
            curr = curr.next
        if saliency:
            # check if saliency is achieved everywhere but missed/abort trials
            # side_out is more complicated
            if simple and event in ["center_in", "side_out"]:
                assert len(event_nodes) <= np.sum(
                    (~self.exp_complexity) & (~self.struct_complexity)
                )
            else:
                assert len(event_nodes) <= self.trialN
        return event_nodes

    def get_event_times(self, event, simple=True, saliency=True):
        """Takes in event and returns the requested event times and their corresponding trial
        Scenarios are exactly as above.
        :param event:
        :param simple:
        :param saliency:
        :return: trial: trial_index simplified from the 0.5 notation
        """
        if isinstance(event, np.ndarray):
            event_nodes = event
        else:
            event_nodes = self.get_event_nodes(event, simple, saliency)
        event_times = np.empty(len(event_nodes), dtype=float)
        trials = np.empty(len(event_nodes), dtype=np.int)
        for ien, enode in enumerate(event_nodes):
            event_times[ien], trials[ien] = enode.etime, enode.trial_index()
        # TODO: for non-salient events, be more careful in handling, be sure to use trials smartly
        return event_times, trials

    def get_trial_event_features(self, feature):
        """Take in feature and return trial features
        feature & event query is mutually dependent, yet we build an abstraction such that the query of
        features seems independent from events. In this manner, 1. for different dataset we only need to
        change the BehaviorMat structure. 2. We could easily chain multiple event features together
        raw feature (as array)
        trial-level feature: (length = trialN)
            OLAT: outcome laterality: -> self.choice_sides (LT/RT) if rel: (IP/CT)
            RW: outcome reward status -> CR/UR
            OTC: outcome status -> same as saliency CR/CU/IU
            ITI family:
                MVT_full: full movement times
                ITI_full: full ITI for decay modeling
                MVT: movement times just for vigor modelling
        event-level feature:
            {event}_MLAT: depending on the simplicity & saliency (MLAT_sal_simp/MLAT_sal/MLAT)

        To get simple unrewarded trials simply do:
        rews = self.get_trial_event_features('RW')
        simp = self.get_trial_event_features('SMP')
        simp_unrew = (rews == 'UR') & (simp != '')
        :param feature:
        :return:
        """
        if "rel" in feature:
            side_map = {
                "left": "IP" if (self.hemisphere == "left") else "CT",
                "right": "CT" if (self.hemisphere == "left") else "IP",
            }
        else:
            side_map = {"left": "LT", "right": "RT"}
        features, trials = None, None

        if "OLAT" in feature:
            features = np.array([side_map[s] for s in self.choice_sides])
            trials = np.arange(self.trialN)
        elif "RW" in feature:
            otcnodes = self.get_event_nodes("outcome", False, False)
            omap = {
                "correct_rewarded": "CR",
                "correct_unrewarded": "CU",
                "incorrect_unrewarded": "IU",
                "missed": "",
                "abort": "",
            }
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif "OTC" in feature:
            otcnodes = self.get_event_nodes("outcome", False, False)
            omap = {
                "correct_rewarded": "CR",
                "correct_unrewarded": "UR",
                "incorrect_unrewarded": "UR",
                "missed": "",
                "abort": "",
            }
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif "SMP" in feature:  # STRUCT or EXPL
            features = np.full(self.trialN, "", dtype=f"<U7")
            features[self.exp_complexity] = "EXPL"
            features[self.struct_complexity] = "STRUCT"
            trials = np.arange(self.trialN)
        elif ("MVT" in feature) or ("ITI" in feature):
            features = self.get_inter_trial_stats(feature)
            trials = np.arange(self.trialN)
        elif "MLAT" in feature:
            feature_args = feature.split("_")
            evt = feature_args[0]
            assert evt != "MLAT", "must have an event option"
            sal = "sal" in feature_args
            simp = ("sal" in feature_args) and ("simp" in feature_args)
            event_nodes = self.get_event_nodes(evt, simp, sal)
            features = [None] * len(event_nodes)
            trials = [0] * len(event_nodes)
            for ien, evn in enumerate(event_nodes):
                features[ien] = evn.mvmt_dynamic()
                trials[ien] = evn.trial_index()
            features = np.array(features)
            trials = np.array(trials)
        else:
            raise NotImplementedError(f"Unknown feature {feature}")
        assert len(features) == len(trials), "weird mismatch"
        # TODO: return data as pd.DataFrame
        return features, trials

    def get_inter_trial_stats(self, option="MVT"):
        """
        :param option:
            'ITI_full': full ITI for decay
            'MVT_full': movement times (whole vigor)
            'MVT': movement times (pure vigor)
        :return:
        """
        side_out_firsts, _ = self.get_event_times("side_out__first", False, True)
        initiates, _ = self.get_event_times("center_in", False, True)
        outcomes, _ = self.get_event_times("outcome", False, True)
        #
        if option == "MVT_full":
            results = initiates - side_out_firsts
        elif option == "ITI_full":
            results = np.zeros(self.trialN)
            results[1:] = initiates[1:] - outcomes[:-1]
        else:
            raise NotImplementedError(f"{option} not implemented")
        return results
