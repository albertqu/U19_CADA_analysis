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


def get_trial_features(mat, feature, as_array=False, drop_empty=True, as_df=False):
    """ OLAT{t-1,t}, RW{t-1,t}, side_out_MLAT_sal{t-1,t}
    for trial level feature or salient MLAT:
        directly use mat trial_feature, get values in corresponding array and then do temporal shift
        accordingly
    all MLAT:
        assert no lag notation and return trial index and corresponding array
    if not as array, convert to dict

    To check what different features this contain, simply return all the keys for the dict/np.unique for
    array option
    :param mat:
    :param feature:
    :param as_array:
    :return:
    """
    if not isinstance(mat, BehaviorMat):
        return np.arange(get_trial_num(mat)), get_trial_features_old(mat, feature, as_array)
    fpast = trial_vector_time_lag
    features = feature.replace(" ", "")
    arg_feature = feature
    feature = feature.split("{")[0]
    if ('MLAT' not in features) or ('MLAT_sal' in features):  # trial level features
        # salient MLAT also considered as trial level
        lags = event_parse_lags(features)
        efeatures, etrials = mat.get_trial_event_features(feature)
        if len(etrials) != mat.trialN:
            assert len(np.unique(etrials)) == len(etrials), 'duplicates contained in salient only?'
            efeatures_temp = np.full_like(efeatures, "")
            efeatures_temp[etrials] = efeatures
            efeatures = efeatures_temp
            etrials = np.arange(mat.trialN)
        maxlen = max([len(ef) for ef in efeatures])
        # does not support ITI for now
        trial_event_features = np.full(len(etrials), "", dtype=f'<U{(maxlen+1) * len(lags)}')
        all_lag_features = [fpast(efeatures, ilag) for ilag in lags]
        for i in range(len(trial_event_features)):
            ith_features = [all_lag_features[il][i] for il in range(len(all_lag_features))]
            if "" in ith_features:
                trial_event_features[i] = ""
            else:
                trial_event_features[i] = "_".join(ith_features)
    else:
        assert '{' not in arg_feature, f'{arg_feature} does not support lag indexing'
        trial_event_features, etrials = mat.get_trial_event_features(feature)

    # Implement ITI bin
    if as_array:
        if as_df:
            return pd.DataFrame({'animal': np.full(len(etrials), mat.animal),
                                 'session': np.full(len(etrials), mat.session),
                                 'behavior_times': trial_event_features, 'trial': etrials})
        return trial_event_features, etrials

    #results = None
    if isinstance(trial_event_features[0], str):
        removal = [""] if drop_empty else []
        results = {feat: (trial_event_features == feat) for feat in np.unique(trial_event_features)
                   if feat not in removal}
    else:
        raise NotImplementedError(f"{feature} (subset of ITI family) not implemented")

    if as_df:
        to_return = pd.DataFrame(results)
        to_return['animal'] = mat.animal
        to_return['session'] = mat.session
        return to_return
    return results, etrials


def get_trial_features_old(mat, feature, as_array=False):
    """
    :param mat:
    # {} syntax for selections of features
    :param feature: time lag coded as {t-k,...} (no space in between allowed)
    :param array_opt: 0 for boolean, 1 for return string array, 2 for digits
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
    elif feature.startswith('A{'):
        feature = feature.replace(" ", "")
        lags = event_parse_lags(feature)
        feature = feature.split("{")[0]
        trial_laterality = get_trial_outcome_laterality(mat)
        lateralities = ('ipsi', 'contra')
        assert len(lags) == 2 and lags[0] < lags[1], 'Other lag so far not implemented'
        for il in lateralities:
            for jl in lateralities:
                stay = 'stay' if (il == jl) else 'switch'
                results[jl + '_' + stay] = np.logical_and.reduce([fpast(trial_laterality[il], lags[1]),
                                                                  fpast(trial_laterality[jl], lags[0])])
    elif feature == 'A{t-1,t}':
        trial_laterality = get_trial_outcome_laterality(mat)
        lateralities = ('ipsi', 'contra')
        for il in lateralities:
            for jl in lateralities:
                stay = 'stay' if (il == jl) else 'switch'
                results[jl + '_' + stay] = np.logical_and.reduce([fpast(trial_laterality[il], 0),
                                                                  fpast(trial_laterality[jl], -1)])
    elif feature.startswith('S['):
        # TODO: extend to ipsi contra
        step = int(feature[2:-1])
        if step > 0:
            sgn = 1
            prepost = '{} Pre'
            op = '+'
        else:
            sgn = -1
            prepost = '{} Post'
            op = '-'
        results = {}
        for i in range(0, step * sgn + 1):
            if i == 0:
                t0s, t1s = 't-1', 't'
            else:
                t0 = -1 + i*sgn
                t1 = i * sgn
                t0s = f't{op}{abs(t0)}' if t0 else 't'
                t1s = f't{op}{i}'

            temp = get_trial_features(mat, 'A{%s,%s}'% (t0s, t1s))
            results[prepost.format(i)] = temp['ipsi_switch'] | temp['contra_switch']

    elif feature == 'ITI':
        itis = access_mat_with_path(mat, "glml/trials/ITI", ravel=True)
        intervals = [(1.05, 4), (0.65, 1.05), (0.5, 0.65), (0, 0.5)]
        results = {itvl: (itis > itvl[0]) & (itis <= itvl[1]) for itvl in intervals}

    elif feature == 'O':
        trial_outcomes = get_trial_outcomes(mat)
        outcomes = ['Incorrect', 'Correct Omission', 'Rewarded']
        results = {oo: trial_outcomes[oo] for oo in outcomes}

    elif feature == 'R':
        results = {o: get_trial_outcomes(mat)[o] for o in ['Unrewarded', 'Rewarded']}

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
        if not isinstance(list(results.keys())[0], str):
            temp = {}
            for rr in results:
                temp[str(rr)] = results[rr]
            results = temp
        maxlen = len(max(results.keys(), key=len))
        feat_array = np.full(N_trial, '', dtype=f'<U{maxlen}')
        for rf in results:
            if len(rf) > 20:
                print("Warning! length greater than 20, string will be truncated")
            feat_array[results[rf]] = rf
        return feat_array
    return results


def get_trial_num(mat):
    if 'glml' in mat:
        mat = access_mat_with_path(mat, "glml", raw=True)
    return int(np.prod(access_mat_with_path(mat, 'trials/ITI').shape))


def decode_trial_behavior(arr, code):
    return {c: arr == code[c] for c in code}


def vectorize_with_map(strvec, vmap):
    res = np.full(len(strvec), np.nan)
    for v in vmap:
        res[strvec == v] = vmap[v]
    return res


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
    """ Takes in vector and shift it by t (pad with False, "" or nan in according to data dtype)
    :param vec: input vector (number, str or bool)
    :param t: shift lag (integer)
    :return: oarr: np.ndarray: shifted array
    @test
    """
    if t == 0:
        return vec
    dtype = vec.dtype
    if np.issubdtype(dtype, np.bool_):
        oarr = np.zeros(len(vec), dtype=dtype)
    elif np.issubdtype(dtype, np.number):
        oarr = np.full(len(vec), np.nan, dtype=np.float)
    elif np.issubdtype(dtype, np.str_):
        oarr = np.full(len(vec), "", dtype=dtype)
    else:
        raise NotImplementedError(f"Unhandled dtype {dtype}")
    if t < 0:
        oarr[-t:] = vec[:t]
    else:
        oarr[:-t] = vec[t:]
    return oarr


def get_behavior_times_old(mat, behavior):
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


def get_behavior_times(mat, behavior, simple=True, saliency=True, as_df=False):
    """ Takes in behavior{t-k} or behavior,
    :param mat: Behavior Mat
    :param behavior: str for behavior events, use {t-k} to zoom in time lags
    :param lag:
    :return: (s x K) where s is determined by lag or behavior arguments
    """
    assert isinstance(mat, BehaviorMat), 'convert to BehaviorMat for performance'
    behavior = behavior.replace(" ", "")
    if not saliency:
        assert '{' not in behavior, 'time shifting in undefined for non-salient events'
    lags = event_parse_lags(behavior)
    assert len(lags) == 1, 'Higher order not implemented'
    behavior = behavior.split("{")[0]
    # TODO: move simple to trial_features
    behavior_times, behavior_trials = mat.get_event_times(behavior, simple, saliency)

    if saliency:
        # Operation only needed for temporal lag
        behavior_temp = np.full(mat.trialN, np.nan)
        behavior_temp[behavior_trials] = behavior_times
        behavior_temp = trial_vector_time_lag(behavior_temp, lags[0])
        nonans = ~np.isnan(behavior_temp)
        behavior_trials, behavior_times = np.arange(mat.trialN)[nonans], behavior_temp[nonans]
    if as_df:
        return pd.DataFrame({'animal': np.full(len(behavior_times), mat.animal),
                             'session': np.full(len(behavior_times), mat.session),
                             'behavior_times': behavior_times, 'trial': behavior_trials})
    return behavior_times, behavior_trials


def map_feature_to_alias(features, maps, old_header):
    features[old_header+'_old'] = features[old_header]
    new_header = old_header
    old_header = old_header+'_old'
    #print(new_header, old_header)
    for mm in maps:
        features[new_header][features[old_header] == mm] = maps[mm]
    return features


def get_correct_port_side_feature(mat):
    portside = np.array(mat['glml/value/cue_port_side'])[:, 0]
    hemi = np.array(mat['glml/notes/hemisphere']).item()
    portside[portside == 2] = 0
    res = np.full(len(portside), 'contra')
    res[portside == hemi] = 'ipsi'
    return res


def get_animal_session_behavior_dataframe(folder, animal, session):
    files = encode_to_filename(folder, animal, session, ['green', 'red', 'FP', 'behavior_old', 'processed'])
    mat = h5py.File(files['behavior_old'], 'r')
    behaviors = ('center_in', 'center_out', 'choice', 'outcome', 'side_out')
    behavior_times = {b: get_behavior_times_old(mat, b)[0] for b in behaviors}
    behavior_pdf = pd.DataFrame(behavior_times)
    fmaps = {'R': {'Rewarded': 'R', 'Unrewarded': 'U', '': ''}}
    rew_feature = get_trial_features_old(mat, 'R', as_array=True)
    side_feature = get_trial_features_old(mat, 'A', as_array=True)
    #     data = np.vstack([trial_vector_time_lag(rew_feature, -2), trial_vector_time_lag(rew_feature, -1),
    #                       trial_vector_time_lag(side_feature, -2), trial_vector_time_lag(side_feature, -1)]).T
    feature_mat = pd.DataFrame(np.vstack([rew_feature, side_feature]).T, columns=['R', 'A'])
    feature_mat['R'] = rew_feature
    feature_mat['A'] = side_feature
    feature_mat = map_feature_to_alias(feature_mat, fmaps['R'], 'R')
    cps = get_correct_port_side_feature(mat)
    feature_mat['C'] = cps
    switch_inds = np.full(len(feature_mat), False)
    switch_inds[1:] = cps[1:] != cps[:-1]
    block_number = np.full(len(feature_mat), 0)
    for i in range(1, len(switch_inds)):
        if not switch_inds[i]:
            block_number[i] = block_number[i-1]+1
    feature_mat['block_num'] = block_number
    header_mat = pd.DataFrame({'trial': np.arange(get_trial_num(mat))})
    header_mat['animal'], header_mat['session'] = animal, session
    header_mat['hemi'] = 'right' if np.array(mat["glml/notes/hemisphere"]).item() else 'left'
    header_mat['region'] = 'NAc' if np.array(mat['glml/notes/region']) else 'DMS'
    return pd.concat([header_mat, behavior_pdf, feature_mat], axis=1)


#######################################################
################### Data Structure ####################
#######################################################
class BehaviorMat:
    # Figure out how to make it general

    code_map = {1: ('center_in', 'center_in'),
                11: ('center_in', 'initiate'),
                2: ('center_out', 'center_out'),
                3: ('side_in', 'left'),
                4: ('side_out', 'left'),
                44: ('side_out', 'left'),
                5: ('side_in', 'right'),
                6: ('side_out', 'right'),
                66: ('side_out', 'right'),
                71.1: ('outcome', 'correct_unrewarded'),
                71.2: ('outcome', 'correct_rewarded'),
                72: ('outcome', 'incorrect_unrewarded'),
                73: ('outcome', 'missed'),  # saliency questionable
                74: ('outcome', 'abort')}  # saliency questionable

    # Always use efficient coding
    def __init__(self, animal, session, hfile, tau=np.inf):
        self.tau = tau
        self.animal = animal
        self.session = session
        if isinstance(hfile, str):
            print("For pipeline loaded hdf5 is recommended for performance")
            hfile = h5py.File(hfile, 'r')
        self.choice_sides = None
        self.exp_complexity = None  # Whether the ITI is complex (first round only analysis simple trials)
        self.struct_complexity = None
        self.trialN = 0
        self.event_list = EventNode(None, None, None, None)
        self.initialize(hfile)

    def __str__(self):
        return f"BehaviorMat({self.animal}_{self.session}, tau={self.tau})"

    def initialize(self, hfile):
        # TODO: reimplement for chris version
        # out.trial_event_mat = trial_event_mat;
        # counted_trial = exper.odor_2afc.param.countedtrial.value;
        # out.outcome = exper.odor_2afc.param.result.value(1:counted_trial);
        # out.port_side = exper.odor_2afc.param.port_side.value(1:counted_trial);
        # out.cue_port_side = exper.odor_2afc.param.cue_port_side.value(1:counted_trial);
        # out.exper_LV_time = Expert_LV_on_time;
        # out.digital_LV_time = LV1_on_time;
        trialN = len(hfile['out/outcome'])
        self.trialN = trialN
        self.choice_sides = np.full(trialN, '', dtype='<U6')
        self.exp_complexity = np.full(trialN, True, dtype=bool)  # default true detect back to back
        self.struct_complexity = np.full(trialN, False, dtype=bool)  # default false detect double centers
        self.exp_complexity[0] = False # TODO: decide where it is fair to ignore exploration before first trial
#         dup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
#                'missed': 0, 'abort': 0}
#         ndup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
#                'missed': 0, 'abort': 0}
#         self.struct_complexity[0] = False
        trial_event_mat = np.array(hfile['out/itrial_event_mat'])

        # Parsing LinkedList
        prev_node = None
        # TODO: Careful of the 0.5 trial events
        for i in range(trial_event_mat.shape[0]):
            eventcode, etime, trial = trial_event_mat[i, :]
            oec = eventcode
            if eventcode == 44 or eventcode == 66:
                eventcode = eventcode // 10
            ctrial = int(np.ceil(trial))-1
            event, opt = BehaviorMat.code_map[eventcode]
            makenew = True

            # for nodes after the first
            if prev_node is not None:
                if eventcode > 70:
                    # for outcome nodes, place laterality as the choice node laterality
                    lat = prev_node.MLAT if eventcode < 73 else ""
                    self.choice_sides[ctrial] = lat
                    if prev_node.event == 'side_in':
                        prev_node.saliency = 'choice'
                if prev_node.etime == etime:
                    if eventcode == prev_node.ecode:
                        makenew = False
                    elif eventcode < 70:
                        print(f"Warning! Duplicate timestamps({prev_node.ecode}, {eventcode}) in {str(self)}")
                    elif eventcode != 72:
                        print(f"Special Event Duplicate: {self.animal}, {self.session}, ", event, opt)
                elif eventcode == 72:
                    print(f"Unexpected non-duplicate for {trial}, {opt}, {self.animal}, {self.session}")
            else:
                assert eventcode < 70, 'outcome cannot be the first node'

            if makenew:
                # potentially fill out all properties here; then make merge an inheriting process
                evnode = self.event_list.append(event, etime, trial, eventcode)
                # Filling MLAT for side ports, Saliency for outcome and initiate
                if event == 'outcome':
                    assert self.choice_sides[ctrial] == prev_node.MLAT
                    evnode.MLAT = prev_node.MLAT
                if eventcode > 6:
                    evnode.saliency = opt
                elif eventcode > 2:
                    evnode.MLAT = opt
                if (oec == 44) or (oec == 66):
                    evnode.saliency = 'execution'
                prev_node = evnode

        # temporal adjacency merge
        assert not self.event_list.is_empty()
        curr_node = self.event_list.next
        while not curr_node.sentinel:
            if '_out' in curr_node.event:
                # COULD do an inner loop to make it look more straightforward
                next_node = curr_node.next
                prev_check = curr_node.prev
                if next_node.sentinel:
                    print(f"Weird early termination with port_out?! {str(curr_node)}")
                # TODO: sanity check: choice side_in does not have any mergeable port before them.
                if (next_node.ecode == curr_node.ecode-1) and (next_node.etime - curr_node.etime < self.tau):
                    merge_node = next_node.next
                    if merge_node.sentinel:
                        print(f"Weird early termination with port_in?! {str(next_node)}")
                    assert merge_node.ecode == curr_node.ecode, f"side in results in {str(merge_node)}"
                    merge_node.merged = True
                    self.event_list.remove_node(curr_node)
                    self.event_list.remove_node(next_node)
                    assert prev_check.next is merge_node and merge_node.prev is prev_check, "Data Structure BUG"
                    curr_node = prev_check  # jump back to previous node

            # Mark features so far saliency: only choice/outcome/initiate, MLAT: outcome/side_port
            if not curr_node.next.merged:  # only trigger at "boundary events" (no new merge happened)
                # Make sure this is not a revisit due to merge
                prev_node = curr_node.prev
                next_node = curr_node.next
                if curr_node.event == 'center_in':
                    # just need MLAT
                    if prev_node.event == 'side_out':
                        curr_node.MLAT = prev_node.MLAT
                    # update structural complexity
                    if curr_node.saliency == 'initiate':
                        breakflag = False
                        cursor = curr_node.prev
                        while (not cursor.sentinel) and (cursor.event != 'outcome'):
                            if cursor.event == 'center_in':
                                self.struct_complexity[curr_node.trial_index()] = True
                                breakflag = True
                                break
                            cursor = cursor.prev
                        if not breakflag and cursor.MLAT:
                            assert cursor.sentinel or (cursor.next.event == 'side_out'), f"weird {cursor}, {cursor.next}"
                elif curr_node.event == 'center_out':
                    if next_node.event == 'side_in':
                        curr_node.MLAT = next_node.MLAT
                    if next_node.saliency == 'choice':
                        # assume "execution" is at center_out, recognizing that well trained animal might
                        # already have executed a program from side_out (denote side port using first/last)
                        curr_node.saliency = 'execution'
                elif curr_node.event == 'side_out':
                    sals = []
                    # TODO: with different TAU we might not want the first side out as salient event
                    if prev_node.event == 'outcome':
                        sals.append('first')
                    if next_node.event == 'center_in':
                        safe_last = True
                        cursor = next_node
                        while cursor.saliency != 'initiate':
                            if cursor.sentinel:
                                print(f"Weird early termination?! {str(cursor.prev)}")
                            if cursor.event == 'side_in':
                                safe_last = False
                                break
                            cursor = cursor.next
                        if safe_last:
                            sals.append('last')
                    curr_node.saliency = "_".join(sals)
                    if len(sals) == 2:
                        self.exp_complexity[int(curr_node.trial)] = False
            curr_node = curr_node.next

    def todf(self):
        return pd.DataFrame({})

    def get_event_nodes(self, event, simple=True, saliency=True):
        # TODO: replace maybe with a DataFrame implementation
        """ Takes in event and returns the requested event nodes
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
        if saliency and 'side_out' in event:
            event, sals = event.split("__")
            if sals == '':
                sals = ['first_last']
                assert simple, "no specific saliency specified for side_out, assume simple trial"
            else:
                sals = [sals, 'first_last']
        else:
            salmap = {'center_in': 'initiate',
                      'center_out': 'execution',
                      'side_in': 'choice',
                      'outcome': ['correct_unrewarded', 'correct_rewarded', 'incorrect_unrewarded']}
            sals = salmap[event]

        while not curr.sentinel:
            if curr.event == event:
                complex_ck = True  # flag for passing the complexity check (irrelevant if simple==False)
                cti = curr.trial_index()
                if simple and event in ['center_in', 'side_out'] and \
                        (self.exp_complexity[cti] or self.struct_complexity[cti]):
                    complex_ck = False

                if ((not saliency) or (curr.saliency != "" and curr.saliency in sals)) and complex_ck:
                    event_nodes.append(curr)
            curr = curr.next
        if saliency:
            # check if saliency is achieved everywhere but missed/abort trials
            # side_out is more complicated
            if simple and event in ['center_in', 'side_out']:
                assert len(event_nodes) <= np.sum((~self.exp_complexity) & (~self.struct_complexity))
            else:
                assert len(event_nodes) <= self.trialN
        return event_nodes

    def get_event_times(self, event, simple=True, saliency=True):
        """ Takes in event and returns the requested event times and their corresponding trial
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
        event_times = np.empty(len(event_nodes), dtype=np.float)
        trials = np.empty(len(event_nodes), dtype=np.int)
        for ien, enode in enumerate(event_nodes):
            event_times[ien], trials[ien] = enode.etime, enode.trial_index()
        # TODO: for non-salient events, be more careful in handling, be sure to use trials smartly
        return event_times, trials

    def get_trial_event_features(self, feature):
        """ Take in feature and return trial features
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
        if 'rel' in feature:
            side_map = {'left': 'IP' if (self.hemisphere == 'left') else 'CT',
                        'right': 'CT' if (self.hemisphere == 'left') else 'IP'}
        else:
            side_map = {'left': 'LT', 'right': 'RT'}
        features, trials = None, None

        if 'OLAT' in feature:
            features = np.array([side_map[s] for s in self.choice_sides])
            trials = np.arange(self.trialN)
        elif 'RW' in feature:
            otcnodes = self.get_event_nodes('outcome', False, False)
            omap = {'correct_rewarded': 'CR', 'correct_unrewarded': 'CU', 'incorrect_unrewarded': 'IU',
                    'missed': '', 'abort': ''}
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif 'OTC' in feature:
            otcnodes = self.get_event_nodes('outcome', False, False)
            omap = {'correct_rewarded': 'CR', 'correct_unrewarded': 'UR', 'incorrect_unrewarded': 'UR',
                    'missed': '', 'abort': ''}
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif 'SMP' in feature: # STRUCT or EXPL
            features = np.full(self.trialN, '', dtype=f'<U7')
            features[self.exp_complexity] = 'EXPL'
            features[self.struct_complexity] = 'STRUCT'
            trials = np.arange(self.trialN)
        elif ('MVT' in feature) or ('ITI' in feature):
            features = self.get_inter_trial_stats(feature)
            trials = np.arange(self.trialN)
        elif 'MLAT' in feature:
            feature_args = feature.split("_")
            evt = feature_args[0]
            assert evt != 'MLAT', 'must have an event option'
            sal = 'sal' in feature_args
            simp = ('sal' in feature_args) and ('simp' in feature_args)
            event_nodes = self.get_event_nodes(evt, simp, sal)
            features = [None] * len(event_nodes)
            trials = [0] * len(event_nodes)
            for ien, evn in enumerate(event_nodes):
                features[ien] = evn.mvmt_dynamic()
                trials[ien] = evn.trial_index()
            features = np.array(features)
            trials = np.array(trials)
        else:
            raise NotImplementedError(f'Unknown feature {feature}')
        assert len(features) == len(trials), 'weird mismatch'
        # TODO: return data as pd.DataFrame
        return features, trials

    def get_inter_trial_stats(self, option='MVT'):
        """
        :param option:
            'ITI_full': full ITI for decay
            'MVT_full': movement times (whole vigor)
            'MVT': movement times (pure vigor)
        :return:
        """
        side_out_firsts, _ = self.get_event_times('side_out__first', False, True)
        initiates, _ = self.get_event_times('center_in', False, True)
        outcomes, _ = self.get_event_times('outcome', False, True)
        #
        if option == 'MVT_full':
            results = initiates - side_out_firsts
        elif option == 'ITI_full':
            results = np.zeros(self.trialN)
            results[1:] = initiates[1:] - outcomes[:-1]
        else:
            raise NotImplementedError(f"{option} not implemented")
        return results


class BehaviorMatChris(BehaviorMat):
    # Figure out how to make it general

    code_map = {1: ('center_in', 'center_in'),
                11: ('center_in', 'initiate'),
                2: ('center_out', 'center_out'),
                3: ('side_in', 'left'),
                4: ('side_out', 'left'),
                44: ('side_out', 'left'),
                5: ('side_in', 'right'),
                6: ('side_out', 'right'),
                66: ('side_out', 'right'),
                71.1: ('outcome', 'correct_unrewarded'),
                71.2: ('outcome', 'correct_rewarded'),
                72: ('outcome', 'incorrect_unrewarded'),
                73: ('outcome', 'missed'),  # saliency questionable
                74: ('outcome', 'abort')}  # saliency questionable

    # Always use efficient coding
    def __init__(self, animal, session, hfile, tau=np.inf):
        self.tau = tau
        self.animal = animal
        self.session = session
        if isinstance(hfile, str):
            print("For pipeline loaded hdf5 is recommended for performance")
            hfile = h5py.File(hfile, 'r')
        self.choice_sides = None
        self.exp_complexity = None  # Whether the ITI is complex (first round only analysis simple trials)
        self.struct_complexity = None
        self.trialN = 0
        self.hemisphere, self.region = None, None
        self.event_list = EventNode(None, None, None, None)
        self.initialize(hfile)

    def __str__(self):
        return f"BehaviorMat({self.animal}_{self.session}, tau={self.tau})"

    def initialize(self, hfile):
        # TODO: reimplement for chris version
        self.hemisphere = 'right' if np.array(hfile["out/notes/hemisphere"]).item() else 'left'
        self.region = 'NAc' if np.array(hfile['out/notes/region']).item() else 'DMS'
        trialN = len(hfile['out/value/outcome'])
        self.trialN = trialN
        self.choice_sides = np.full(trialN, '', dtype='<U6')
        self.exp_complexity = np.full(trialN, True, dtype=bool)  # default true detect back to back
        self.struct_complexity = np.full(trialN, False, dtype=bool)  # default false detect double centers
        self.exp_complexity[0] = False
#         dup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
#                'missed': 0, 'abort': 0}
#         ndup = {'correct_unrewarded': 0, 'correct_rewarded': 0, 'incorrect_unrewarded': 0,
#                'missed': 0, 'abort': 0}
#         self.struct_complexity[0] = False
        trial_event_mat = np.array(hfile['out/value/trial_event_mat'])

        # Parsing LinkedList
        prev_node = None
        # TODO: Careful of the 0.5 trial events
        for i in range(trial_event_mat.shape[0]):
            eventcode, etime, trial = trial_event_mat[i, :]
            if eventcode == 44 or eventcode == 66:
                eventcode = eventcode // 10
            ctrial = int(np.ceil(trial))-1
            event, opt = BehaviorMat.code_map[eventcode]
            makenew = True

            if prev_node is not None:
                if eventcode > 70:
                    lat = prev_node.MLAT if eventcode < 73 else ""
                    self.choice_sides[ctrial] = lat
                    if prev_node.event == 'side_in':
                        prev_node.saliency = 'choice'
                if prev_node.etime == etime:
                    if eventcode == prev_node.ecode:
                        makenew = False
                    elif eventcode < 70:
                        print(f"Warning! Duplicate timestamps({prev_node.ecode}, {eventcode}) in {str(self)}")
                    elif eventcode != 72:
                        print(f"Special Event Duplicate: {self.animal}, {self.session}, ", event, opt)
                elif eventcode == 72:
                    print(f"Unexpected non-duplicate for {trial}, {opt}, {self.animal}, {self.session}")
            else:
                assert eventcode < 70, 'outcome cannot be the first node'

            if makenew:
                # potentially fill out all properties here; then make merge an inheriting process
                evnode = self.event_list.append(event, etime, trial, eventcode)
                # Filling MLAT for side ports, Saliency for outcome and initiate
                if event == 'outcome':
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
            if '_out' in curr_node.event:
                # COULD do an inner loop to make it look more straightforward
                next_node = curr_node.next
                prev_check = curr_node.prev
                if next_node.sentinel:
                    print(f"Weird early termination with port_out?! {str(curr_node)}")
                # TODO: sanity check: choice side_in does not have any mergeable port before them.
                if (next_node.ecode == curr_node.ecode-1) and (next_node.etime - curr_node.etime < self.tau):
                    merge_node = next_node.next
                    if merge_node.sentinel:
                        print(f"Weird early termination with port_in?! {str(next_node)}")
                    assert merge_node.ecode == curr_node.ecode, f"side in results in {str(merge_node)}"
                    merge_node.merged = True
                    self.event_list.remove_node(curr_node)
                    self.event_list.remove_node(next_node)
                    assert prev_check.next is merge_node and merge_node.prev is prev_check, "Data Structure BUG"
                    curr_node = prev_check  # jump back to previous node

            # Mark features so far saliency: only choice/outcome/initiate, MLAT: outcome/side_port
            if not curr_node.next.merged:  # only trigger at "boundary events" (no new merge happened)
                # Make sure this is not a revisit due to merge
                prev_node = curr_node.prev
                next_node = curr_node.next
                if curr_node.event == 'center_in':
                    # just need MLAT
                    if prev_node.event == 'side_out':
                        curr_node.MLAT = prev_node.MLAT
                    # update structural complexity
                    if curr_node.saliency == 'initiate':
                        breakflag = False
                        cursor = curr_node.prev
                        while (not cursor.sentinel) and (cursor.event != 'outcome'):
                            if cursor.event == 'center_in':
                                self.struct_complexity[curr_node.trial_index()] = True
                                breakflag = True
                                break
                            cursor = cursor.prev
                        if not breakflag and cursor.MLAT:
                            assert cursor.sentinel or (cursor.next.event == 'side_out'), f"weird {cursor}, {cursor.next}"
                elif curr_node.event == 'center_out':
                    if next_node.event == 'side_in':
                        curr_node.MLAT = next_node.MLAT
                    if next_node.saliency == 'choice':
                        # assume "execution" is at center_out, recognizing that well trained animal might
                        # already have executed a program from side_out (denote side port using first/last)
                        curr_node.saliency = 'execution'
                elif curr_node.event == 'side_out':
                    sals = []
                    # TODO: with different TAU we might not want the first side out as salient event
                    if prev_node.event == 'outcome':
                        sals.append('first')
                    if next_node.event == 'center_in':
                        safe_last = True
                        cursor = next_node
                        while cursor.saliency != 'initiate':
                            if cursor.sentinel:
                                print(f"Weird early termination?! {str(cursor.prev)}")
                            if cursor.event == 'side_in':
                                safe_last = False
                                break
                            cursor = cursor.next
                        if safe_last:
                            sals.append('last')
                    curr_node.saliency = "_".join(sals)
                    if len(sals) == 2:
                        self.exp_complexity[int(curr_node.trial)] = False
            curr_node = curr_node.next

    def todf(self):
        elist = self.event_list
        if elist.is_empty():
            return None
        fields = ['trial', 'center_in', 'center_out', 'side_in', 'outcome',
                  'side_out', 'ITI', 'A', 'R', 'BLKNo', 'CPort']



    def get_event_nodes(self, event, simple=True, saliency=True):
        # TODO: replace maybe with a DataFrame implementation
        """ Takes in event and returns the requested event nodes
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
        if saliency and 'side_out' in event:
            event, sals = event.split("__")
            if sals == '':
                sals = ['first_last']
                assert simple, "no specific saliency specified for side_out, assume simple trial"
            else:
                sals = [sals, 'first_last']
        else:
            salmap = {'center_in': 'initiate',
                      'center_out': 'execution',
                      'side_in': 'choice',
                      'outcome': ['correct_unrewarded', 'correct_rewarded', 'incorrect_unrewarded']}
            sals = salmap[event]

        while not curr.sentinel:
            if curr.event == event:
                complex_ck = True  # flag for passing the complexity check (irrelevant if simple==False)
                cti = curr.trial_index()
                if simple and event in ['center_in', 'side_out'] and \
                        (self.exp_complexity[cti] or self.struct_complexity[cti]):
                    complex_ck = False

                if ((not saliency) or (curr.saliency != "" and curr.saliency in sals)) and complex_ck:
                    event_nodes.append(curr)
            curr = curr.next
        if saliency:
            # check if saliency is achieved everywhere but missed/abort trials
            # side_out is more complicated
            if simple and event in ['center_in', 'side_out']:
                assert len(event_nodes) <= np.sum((~self.exp_complexity) & (~self.struct_complexity))
            else:
                assert len(event_nodes) <= self.trialN
        return event_nodes

    def get_event_times(self, event, simple=True, saliency=True):
        """ Takes in event and returns the requested event times and their corresponding trial
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
        event_times = np.empty(len(event_nodes), dtype=np.float)
        trials = np.empty(len(event_nodes), dtype=np.int)
        for ien, enode in enumerate(event_nodes):
            event_times[ien], trials[ien] = enode.etime, enode.trial_index()
        # TODO: for non-salient events, be more careful in handling, be sure to use trials smartly
        return event_times, trials

    def get_trial_event_features(self, feature):
        """ Take in feature and return trial features
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
        if 'rel' in feature:
            side_map = {'left': 'IP' if (self.hemisphere == 'left') else 'CT',
                        'right': 'CT' if (self.hemisphere == 'left') else 'IP'}
        else:
            side_map = {'left': 'LT', 'right': 'RT'}
        features, trials = None, None

        if 'OLAT' in feature:
            features = np.array([side_map[s] for s in self.choice_sides])
            trials = np.arange(self.trialN)
        elif 'RW' in feature:
            otcnodes = self.get_event_nodes('outcome', False, False)
            omap = {'correct_rewarded': 'CR', 'correct_unrewarded': 'CU', 'incorrect_unrewarded': 'IU',
                    'missed': '', 'abort': ''}
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif 'OTC' in feature:
            otcnodes = self.get_event_nodes('outcome', False, False)
            omap = {'correct_rewarded': 'CR', 'correct_unrewarded': 'UR', 'incorrect_unrewarded': 'UR',
                    'missed': '', 'abort': ''}
            features = np.array([omap[onode.saliency] for onode in otcnodes])
            trials = np.arange(self.trialN)
        elif 'SMP' in feature: # STRUCT or EXPL
            features = np.full(self.trialN, '', dtype=f'<U7')
            features[self.exp_complexity] = 'EXPL'
            features[self.struct_complexity] = 'STRUCT'
            trials = np.arange(self.trialN)
        elif ('MVT' in feature) or ('ITI' in feature):
            features = self.get_inter_trial_stats(feature)
            trials = np.arange(self.trialN)
        elif 'MLAT' in feature:
            feature_args = feature.split("_")
            evt = feature_args[0]
            assert evt != 'MLAT', 'must have an event option'
            sal = 'sal' in feature_args
            simp = ('sal' in feature_args) and ('simp' in feature_args)
            event_nodes = self.get_event_nodes(evt, simp, sal)
            features = [None] * len(event_nodes)
            trials = [0] * len(event_nodes)
            for ien, evn in enumerate(event_nodes):
                features[ien] = evn.mvmt_dynamic()
                trials[ien] = evn.trial_index()
            features = np.array(features)
            trials = np.array(trials)
        else:
            raise NotImplementedError(f'Unknown feature {feature}')
        assert len(features) == len(trials), 'weird mismatch'
        # TODO: return data as pd.DataFrame
        return features, trials

    def get_inter_trial_stats(self, option='MVT'):
        """
        :param option:
            'ITI_full': full ITI for decay
            'MVT_full': movement times (whole vigor)
            'MVT': movement times (pure vigor)
        :return:
        """
        side_out_firsts, _ = self.get_event_times('side_out__first', False, True)
        initiates, _ = self.get_event_times('center_in', False, True)
        outcomes, _ = self.get_event_times('outcome', False, True)
        #
        if option == 'MVT_full':
            results = initiates - side_out_firsts
        elif option == 'ITI_full':
            results = np.zeros(self.trialN)
            results[1:] = initiates[1:] - outcomes[:-1]
        else:
            raise NotImplementedError(f"{option} not implemented")
        return results


class EventNode:
    ABBR = {
        'right': 'RT',
        'left': 'LT',
        'ipsi': 'IP',
        'contra': 'CT',
        'center': 'CE',
    }

    def __init__(self, event, etime, trial, ecode):
        self.event = event
        self.trial = trial # trial starts at 1 instead of 0
        self.etime = etime
        self.ecode = ecode # For debug purpose
        # Use "" for Null
        self.MLAT = ""
        self.saliency = ""
        self.merged = False
        self.next = None
        self.prev = None
        if event is None:
            # Implements a circular LinkedList
            self.sentinel = True
            self.next = self
            self.prev = self
            self.size = 0
        else:
            self.sentinel = False

    def as_array(self):
        # Returns an array representation of the information
        pass

    def mvmt_dynamic(self):
        """
        center_in/out: LT->CE / RT->CE (IP->CE/CT->CE)
        side_in/out: LT/RT (IP/CT)
        Returns dynamic of movement considering previous node relative to current node
        :return:
        """
        return self.MLAT
        #pass

    def trial_index(self):
        # 0.5 is ITI but considered in trial 0
        return int(np.ceil(self.trial)) - 1

    def __str__(self):
        return f"EventNode({self.event}, {self.trial}, {self.etime:.1f}ms, {self.ecode})"

    # Methods Reserved For Sentinel Node
    def __len__(self):
        assert self.sentinel, 'must be sentinel node to do this'
        return self.size

    # ideally add iter method but not necessary
    def tolist(self):
        assert self.sentinel, 'must be sentinel node to do this'
        cursor = self.next
        result = [None] * len(self)
        i = 0
        while not cursor.sentinel:
            result[i] = cursor
            cursor = cursor.next
            i += 1
        return result

    def append(self, event, etime, trial, ecode):
        assert self.sentinel, 'must be sentinel node to do this'
        evn = EventNode(event, etime, trial, ecode)
        old_end = self.prev
        assert old_end.next is self, "what is happening"
        old_end.next = evn
        evn.prev = old_end
        self.prev = evn
        evn.next = self
        self.size += 1
        return evn

    def prepend(self):
        # Not important
        assert self.sentinel, 'must be sentinel node to do this'
        pass

    def remove_node(self, node):
        assert self.sentinel, 'must be sentinel node to do this'
        assert self.size, 'list must be non-empty'
        next_node = node.next
        prev_node = node.prev
        prev_node.next = next_node
        next_node.prev = prev_node
        node.next = None
        node.prev = None
        self.size -= 1

    def get_last(self):
        assert self.sentinel, 'must be sentinel node to do this'
        return self.prev

    def get_first(self):
        assert self.sentinel, 'must be sentinel node to do this'
        return self.next

    def is_empty(self):
        assert self.sentinel, 'must be sentinel node to do this'
        return self.size == 0







