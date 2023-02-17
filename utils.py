# System
import time, os, h5py, re
import logging
import graphviz
# Structure
from collections import deque
# Data
import scipy
import numpy as np
import pandas as pd
from scipy.sparse import diags as spdiags
from scipy.sparse import linalg as sp_linalg
from scipy import interpolate, signal
from utils_models import auc_roc_2dist
from utils_signal import std_filter, median_filter
from packages.photometry_functions import get_dFF
# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from packages.photometry_functions import get_f0_Martianova_jove, jove_fit_reference, smooth_signal, airPLS
# caiman
try:
    from caiman.source_extraction.cnmf.deconvolution import GetSn
    from caiman.source_extraction.cnmf.utilities import fast_prct_filt
    from caiman.utils.stats import df_percentile
except ModuleNotFoundError:
    print("CaImAn not installed or environment not activated, certain functions might not be usable")

RAND_STATE = 230

# TODO: Move project specific portions to pipeline_*.py as things scale

##################################################
#################### Loading #####################
##################################################


def get_probswitch_session_by_condition(folder, group='all', region='NAc', signal='all'):
    """ Searches through [folder] and find all of probswitch experiment sessions that match the
    description; Returns lists of session files of different recording type
    :param group: str, expression
    :param region: str, region of recording
    :param signal: str, signal type (DA or Ca 05/25/21)
    :param photometry:
    :param choices:
    :param processed:
    :return:
    """
    if group == 'all':
        groups = ('D1', 'A2A')
    else:
        groups = [group]
    if region == 'all':
        regions = ['NAc', 'DMS']
    else:
        regions = [region]
    if signal == 'all':
        signals = ['DA', 'Ca']
    else:
        signals = [signal]
    results = {}
    for g in groups:
        grouppdf = pd.read_csv(os.path.join(folder, f"ProbSwitch_FP_Mice_{g}.csv"))
        rsel = grouppdf['Region'].isin(regions)
        if signals[0] == 'none':
            animal_sessions = grouppdf[rsel]
        else:
            fpsel = grouppdf['FP'] >= 1
            sigsel = np.logical_and.reduce([grouppdf[f'FP_{s}_zoom'] > 0 for s in signals])
            animal_sessions = grouppdf[rsel & fpsel & sigsel]
        results[g] = {}
        for animal in animal_sessions['animal'].unique():
            results[g][animal] = sorted(animal_sessions[animal_sessions['animal'] == animal]['session'])
    return results


def get_prob_switch_all_sessions(folder, groups):
    """ Exhaustively check all folder that contains ProbSwitch task .mat files and encode all sessions.
    .mat -> decode -> return group
    :param folder:
    :return:
    """
    only_Ca = []
    only_DA = []
    results = {g: {a: [] for a in groups[g]} for g in groups}
    for d in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, d)):
            m = re.match("^(?P<animal>\w{2,3}-\d{2,}[-\w*]*_[A-Z]{2})_(?P<session>p\d+\w+)", d)
            if m:
                animal, day = m.group('animal'), m.group('session')
                group_dict = results[animal.split("-")[0]]
                if animal in group_dict:
                    group_dict[animal].append(day)
                elif animal not in group_dict and '*' in group_dict:
                    group_dict[animal] = [day]
    for g in results:
        del results[g]['*']
    return results


def check_FP_contain_dff_method(fp, methods, sig='DA'):
    """ Utility function that helps check whether the <fp> hdf5 file contains <dff> signals preprocessed.
    """
    if fp is None:
        return False
    if isinstance(methods, str):
        methods = [methods]
    with h5py.File(fp, 'r') as hf:
        return np.all([f'{sig}/dff/{m}' in hf for m in methods])


def get_sources_from_csvs(csvfiles, window=400, delim=None, aux_times=None, tags=None, show=False):
    """
    Extract sources from a list of csvfiles, with csvfile[0] be channels with cleaniest
    TODO: potentially use the fact that 415 has the same timestamps to speed up the process
    :param csvfiles:
    :param window:
    :return:
    """
    if isinstance(csvfiles, str):
        csvfiles = [csvfiles]
    if delim is None:
        delim = " "
    try:
        pdf = pd.read_csv(csvfiles[0], delimiter=delim, names=['time', 'calcium'], usecols=[0, 1])
        FP_times = [None] * len(csvfiles)
        FP_signals = [None] * len(csvfiles)
        for i in range(len(csvfiles)):
            csvfile = csvfiles[i]
            # Signal Sorting
            pdf = pd.read_csv(csvfile, delimiter=delim, names=['time', 'calcium'], usecols=[0, 1])
            FP_times[i] = pdf.time.values
            FP_signals[i] = pdf.calcium.values
        if aux_times:
            old_zero = FP_times[0][0]
            if old_zero == aux_times[0][0]:
                print('WARNING: NO UPDATE, something is up')
            assert len(FP_times) == len(aux_times), 'MUST BE SAME dim'
            FP_times = aux_times

        if tags is None:
            tags = [f'REC{i}' for i in range(len(csvfiles))]
    except:
        print('OOPPS')
        # TODO: aux_time input potentially needed
        FP_times = [None] * len(csvfiles) * 2
        FP_signals = [None] * len(csvfiles) * 2
        for i in range(len(csvfiles)):
            # Signal Sorting
            csvfile = csvfiles[i]
            pdf = pd.read_csv(csvfile, delimiter=",")
            red_sig = pdf['Region1R'].values
            green_sig = pdf['Region0G'].values
            times = pdf['Timestamp'].values
            FP_times[2 * i], FP_times[2*i+1] = times, times
            FP_signals[2 * i], FP_signals[2*i+1] = green_sig, red_sig
        if tags is None:
            tags = np.concatenate([[f'REC{i}_G', f'REC{i}_R'] for i in range(len(csvfiles))])

    FP_REC_signals = [None] * len(FP_signals)
    FP_REC_times = [None] * len(FP_signals)
    FP_415_signals = [None] * len(FP_signals)
    FP_415_times = [None] * len(FP_signals)
    FP_415_sel = None
    for i in range(len(FP_signals)):
        FP_time, FP_signal = FP_times[i], FP_signals[i]

        # # Plain Threshold
        # min_signal, max_signal = np.min(FP_signal), np.max(FP_signal)
        # intensity_threshold = min_signal+(max_signal - min_signal)*0.4

        # Dynamic Threshold
        n_win = len(FP_signal) // window
        bulk = n_win * window
        edge = len(FP_signal) - bulk
        first_batch = FP_signal[:bulk].reshape((n_win, window), order='C')
        end_batch = FP_signal[-window:]
        edge_batch = FP_signal[-edge:]
        sigT_sels = np.concatenate([(first_batch > np.mean(first_batch, keepdims=True, axis=1))
                                   .reshape(bulk, order='C'), edge_batch > np.mean(end_batch)])

        sigD_sels = ~sigT_sels
        FP_top_signal, FP_top_time = FP_signal[sigT_sels], FP_time[sigT_sels]
        FP_down_signal, FP_down_time = FP_signal[sigD_sels], FP_time[sigD_sels]
        topN, downN = len(FP_top_signal)//window, len(FP_down_signal)//window
        top_dyn_std = np.std(FP_top_signal[:topN * window].reshape((topN, window),order='C'), axis=1).mean()
        down_dyn_std = np.std(FP_down_signal[:downN * window].reshape((downN,window),order='C'),axis=1).mean()
        # TODO: check for consecutives
        # TODO: check edge case when only 415 has signal
        if top_dyn_std >= down_dyn_std:
            sigREC_sel, sig415_sel = sigT_sels, sigD_sels
            FP_REC_signals[i], FP_REC_times[i] = FP_top_signal, FP_top_time
            FP_415_signals[i], FP_415_times[i] = FP_down_signal, FP_down_time
        else:
            sigREC_sel, sig415_sel = sigD_sels, sigT_sels
            FP_REC_signals[i], FP_REC_times[i] = FP_down_signal, FP_down_time
            FP_415_signals[i], FP_415_times[i] = FP_top_signal, FP_top_time

    if show:
        fig, axes = plt.subplots(nrows=len(FP_REC_signals), ncols=1, sharex=True)
        for i in range(len(FP_REC_signals)):
            ax = axes[i] if len(FP_REC_signals) > 1 else axes
            itag = tags[i]
            ax.plot(FP_REC_times[i], FP_REC_signals[i], label=itag)
            ax.plot(FP_415_times[i], FP_415_signals[i], label='415')
            ax.legend()
    # TODO: save as pd.DataFrame
    if len(FP_REC_signals) == 1:
        return FP_REC_times[0], FP_REC_signals[0], FP_415_times[0], FP_415_signals[0]
    # TODO: if shape uniform merge signals
    return FP_REC_times, FP_REC_signals, FP_415_times, FP_415_signals


def path_prefix_free(path):
    symbol = os.path.sep
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol, 0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol) + len(symbol):]


def file_folder_path(f):
    symbol = os.path.sep
    len_sym = len(symbol)
    if f[-len_sym:] == symbol:
        return f[:f.rfind(symbol, 0, -len_sym)]
    else:
        return f[:f.rfind(symbol)]


def summarize_sessions(data_root, implant_csv, save_path, sort_key='aID'):
    """
    implant_csv: pd.DataFrame from implant csv file
    """
    # add region of implant, session number, signal quality
    # input a list of names implant locations
    # "/A2A-15B-B_RT_20200229_learning-switch-2_p39.mat" supposed to be 139
    # sorting with p notation mess up if p is less 100\
    # bug /D1-27H_LT_20200229_ToneSamp_p89.mat read as 022
    alles = {'animal': [], 'aID':[], 'session': [], 'date': [], 'ftype':[],
             'age':[], 'FP': [], 'region': [], 'note': []}

    implant_lookup = {}
    for i in range(len(implant_csv)):

        animal_name = implant_csv.loc[i, 'Name']
        if animal_name and (str(animal_name) != 'nan'):
            LH_target = implant_csv.loc[i, 'LH Target']
            RH_target = implant_csv.loc[i, 'RH Target']
            print(animal_name)
            name_first, name_sec = animal_name.split(' ')
            name_first = "-".join(name_first.split('-')[:2])
            implant_lookup[name_first+'_'+name_sec] = {'LH': LH_target, 'RH': RH_target}

    for f in os.listdir(data_root):
        options = decode_from_filename(f)
        if options is None:
            pass
            #print(f, "ALERT")
        elif ('FP_' in f) and ('FP_' not in options['session']):
            print(f, options['session'])
        else:
            for q in ['animal', 'ftype', 'session']:
                alles[q].append(options[q])

            name_first2, name_sec2 = options['animal'].split('_')
            name_first2 = "-".join(name_first2.split('-')[:2])
            aID = name_first2+"_"+name_sec2
            alles['aID'].append(aID)
            alles['date'].append(options['T'])
            opts = options['session'].split("_FP_")
            alles['age'].append(opts[0])
            if len(opts) > 1:
                alles['FP'].append(opts[1])
                if aID not in implant_lookup:
                    print('skipping', options, )
                    alles['region'].append('')
                else:
                    alles['region'].append(implant_lookup[aID][opts[1]])
            else:
                alles['FP'].append("")
                alles['region'].append('')
            alles['note'].append(options['DN'] + options['SP'])

    apdf = pd.DataFrame(alles)
    sorted_pdf = apdf.sort_values(['date', 'session'], ascending=True)
    sorted_pdf['S_no'] = 0
    new_pdfs = []
    for anim in sorted_pdf[sort_key].unique():
        tempslice = sorted_pdf[sorted_pdf[sort_key] == anim]
        sorted_pdf.loc[sorted_pdf[sort_key] == anim, 'S_no'] = np.arange(1, len(tempslice)+1)
    #final_pdf = pd.concat(new_pdfs, axis=0)
    final_pdf = sorted_pdf
    final_pdf.to_csv(os.path.join(save_path, f"exper_list_final_{sort_key}.csv"), index=False)


def encode_to_filename(folder, animal, session, ftypes="processed_all"):
    """
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
            if ftypes=="all"
    :return:
            returns all 5 files in a dictionary; otherwise return all file types
            in a dictionary, None if not found
    """
    # TODO: enable aliasing
    paths = [os.path.join(folder, animal, session), os.path.join(folder, animal+'_'+session),
             os.path.join(folder, animal), folder]
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
                opt = decode_from_filename(f)
                if opt is not None:
                    ift = opt['ftype']
                    check_mark = opt['animal'] == animal and opt['session'] == session
                    #print(opt['session'], animal, session)
                    check_mark_mdl = (opt['animal'] == animal) and (opt['session'] in session)
                    cm_mdl = (ift == 'modeling' and check_mark_mdl)
                    # TODO: temporary hacky method for modeling
                    #print(opt['session'], animal, session, check_mark_mdl, ift, cm_mdl)
                    if ift in ftypes and results[ift] is None and (check_mark or cm_mdl):
                        results[ift] = os.path.join(p, f)
                        registers += 1
                        if registers == len(ftypes):
                            return results if len(results) > 1 else results[ift]
    return results if len(results) > 1 else list(results.values())[0]


def decode_from_filename(filename):
    """
    Takes in filenames of the following formats and returns the corresponding file options
    `A2A-15B_RT_20200612_ProbSwitch_p243_FP_RH`, `D1-27H_LT_20200314_ProbSwitch_FP_RH_p103`
    behavioral: * **Gen-ID_EarPoke_Time_DNAME_Age_special.mat**
FP: **Gen-ID_EarPoke_DNAME2_Hemi_Age_channel_Time(dash)[Otherthing].csv**
binary matrix: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_Binary_Matrix)Time[special].etwas**
timestamps: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_timestamps)Time[special].csv**
    GEN: genetic line, ID: animal ID, EP: ear poke, T: time of expr, TD: detailed HMS DN: Data Name, A: Age,
    H: hemisphere, S: session, SP: special extension
    :param filename:
    :return: options: dict
                ftype
                animal
                session
    """
    filename = path_prefix_free(filename)
    # case exper

    mBMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<T>\d+)_(?P<DN>[-&\w]+)_("
                      r"?P<A>p\d+)(?P<SP>[-&\w]*)\.mat", filename)
    # case processed behavior
    mPBMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H)_processed_data.mat", filename)
    mPBOMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H)_behavior_data.mat", filename)
    mFPMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H).hdf5", filename)
    mMDMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>(FP_[LR]H)?)_modeling.hdf5", filename)
    mTBMat =re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)_trialB.csv", filename)

    # case binary
    mBIN = None
    options, ftype = None, None
    if mBMat is not None:
        # TODO: handle session#
        options = mBMat.groupdict()
        ftype = "exper"
        oS = options["SP"]
        options["H"] = ""
        dn_match = re.match(".*(FP_[LR]H).*", options['DN'])
        sp_match = re.match(".*(FP_[LR]H).*", options['SP'])
        if dn_match:
            options["H"] = dn_match.group(1)
        elif sp_match:
            options['H'] = sp_match.group(1)
    elif mTBMat is not None:
        options = mTBMat.groupdict()
        ftype = 'trialB'
        oS = options['S']
    elif mMDMat is not None:
        options = mMDMat.groupdict()
        ftype = 'modeling'
        oS = options['S']
    elif mPBMat is not None:
        options = mPBMat.groupdict()
        ftype = "processed"
        oS = options["S"]
    elif mPBOMat is not None:
        options = mPBOMat.groupdict()
        ftype = "behavior_old"
        oS = options["S"]
    elif mFPMat is not None:
        options = mFPMat.groupdict()
        ftype = "FP"
        oS = options['S']
    elif mBIN is not None:
        # TODO: fill it up
        options = mBIN.groupdict()
        oS = ""
        ftype = "bin_mat"

    else:
        #TODO: print("Warning! Certain sessions have inconsistent naming! needs more through check")
        # case csv
        #todo: merge cage id and earpoke
        channels = ['keystrokes', "MetaData", "NIDAQ_Ai0_timestamp", "NIDAQ_Ai0_Binary_Matrix",
                    "red", "green", "FP", 'FPTS']
        for c in channels:
            mCSV = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<DN>[-&\w]+)_(?P<H>([LR]H_)?)(?P<A>p\d+)(?P<SP>[-&\w]*)" +
                            f"_{c}(?P<BSC>(_(BSC)\d)?)" + r"(?P<S>_session\d+_|_?)(?P<T>\d{4}-?\d{2}-?\d{2})T(?P<TD>[_\d]+)\.[(csv)|\d+]", filename)

            if mCSV is not None:
                options = mCSV.groupdict()
                ftype = c
                oS = options["S"]
                options['H'] = "FP_" + options['H']
                break
                # print(filename)
                # print(options)
        if ftype is None:
            #print("special:", filename)
            return None
    mS = re.match(r".*(session\d+).*", oS)
    fS = ""
    if mS:
        fS = "_"+mS.group(1)
    options["ftype"] = ftype
    options["animal"] = options['GEN'] + "-" + options["ID"] + "_" + options["EP"]
    options["session"] = options['A'] + fS + (("_"+options['H']) if options['H'] else "")
    return options


# Figure out rigorous representation; also keep old version intact
def encode_to_filename_new(folder, animal, session, ftypes="processed_all"):
    """
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
            if ftypes=="all"
    :return:
            returns all 5 files in a dictionary; otherwise return all file types
            in a dictionary, None if not found
    """
    # TODO: enable aliasing
    paths = [os.path.join(folder, animal, session), os.path.join(folder, animal+'_'+session),
             os.path.join(folder, animal), folder]
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
                    if ift == 'FP':
                        ift_arg = 'FP_'
                    else:
                        ift_arg = ift
                    if (ift_arg in f) and (animal in f) and (session in f):
                        results[ift] = os.path.join(p, f)
                        registers += 1
                        if registers == len(ftypes):
                            return results if len(results) > 1 else results[ift]
                # opt = decode_from_filename(f)
                # if opt is not None:
                #     ift = opt['ftype']
                #     check_mark = opt['animal'] == animal and opt['session'] == session
                #     #print(opt['session'], animal, session)
                #     check_mark_mdl = (opt['animal'] == animal) and (opt['session'] in session)
                #     cm_mdl = (ift == 'modeling' and check_mark_mdl)
                #     # TODO: temporary hacky method for modeling
                #     #print(opt['session'], animal, session, check_mark_mdl, ift, cm_mdl)
                #     if ift in ftypes and results[ift] is None and (check_mark or cm_mdl):
                #         results[ift] = os.path.join(p, f)
                #         registers += 1
                #         if registers == len(ftypes):
                #             return results if len(results) > 1 else results[ift]
    return results if len(results) > 1 else list(results.values())[0]


def decode_from_filename_new(filename):
    """
    Takes in filenames of the following formats and returns the corresponding file options
    `A2A-15B_RT_20200612_ProbSwitch_p243_FP_RH`, `D1-27H_LT_20200314_ProbSwitch_FP_RH_p103`
    behavioral: * **Gen-ID_EarPoke_Time_DNAME_Age_special.mat**
FP: **Gen-ID_EarPoke_DNAME2_Hemi_Age_channel_Time(dash)[Otherthing].csv**
binary matrix: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_Binary_Matrix)Time[special].etwas**
timestamps: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_timestamps)Time[special].csv**
    GEN: genetic line, ID: animal ID, EP: ear poke, T: time of expr, TD: detailed HMS DN: Data Name, A: Age,
    H: hemisphere, S: session, SP: special extension
    :param filename:
    :return: options: dict
                ftype
                animal
                session
    """
    filename = path_prefix_free(filename)
    # case exper
    mBMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<T>\d+)_(?P<DN>[-&\w]+)_("
                      r"?P<A>p\d+)(?P<SP>[-&\w]*)\.mat", filename)
    # case processed behavior
    mPBMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H)_processed_data.mat", filename)
    mPBOMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H)_behavior_data.mat", filename)
    mFPMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>FP_[LR]H).hdf5", filename)
    mMDMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>(\d|\w){3,}[-\w*]*)_(?P<EP>[A-Z]{2})_"
                      r"(?P<A>p\d+)(?P<S>_session\d+_|_?)(?P<H>(FP_[LR]H)?)_modeling.hdf5", filename)
    mTBMat = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>(\d|\w){3,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<A>p\d+)(?P<S>(_session\d+)?)_trialB.csv", filename)

    # case binary
    mBIN = None
    options, ftype = None, None
    if mBMat is not None:
        # TODO: handle session#
        options = mBMat.groupdict()
        ftype = "exper"
        oS = options["SP"]
        options["H"] = ""
        dn_match = re.match(".*(FP_[LR]H).*", options['DN'])
        sp_match = re.match(".*(FP_[LR]H).*", options['SP'])
        if dn_match:
            options["H"] = dn_match.group(1)
        elif sp_match:
            options['H'] = sp_match.group(1)
    elif mTBMat is not None:
        options = mTBMat.groupdict()
        ftype = 'trialB'
        oS = options['S']
        options['H'] = ''
    elif mMDMat is not None:
        options = mMDMat.groupdict()
        ftype = 'modeling'
        oS = options['S']
    elif mPBMat is not None:
        options = mPBMat.groupdict()
        ftype = "processed"
        oS = options["S"]
    elif mPBOMat is not None:
        options = mPBOMat.groupdict()
        ftype = "behavior_old"
        oS = options["S"]
    elif mFPMat is not None:
        options = mFPMat.groupdict()
        ftype = "FP"
        oS = options['S']
    elif mBIN is not None:
        # TODO: fill it up
        options = mBIN.groupdict()
        oS = ""
        ftype = "bin_mat"
    else:
        #TODO: print("Warning! Certain sessions have inconsistent naming! needs more through check")
        # case csv
        #todo: merge cage id and earpoke
        """A2A-16B-1_RT_ChR2_switch_no_cue_LH_p147_red_2020-03-17T15_38_40.csv"""
        channels = ['keystrokes', "MetaData", "NIDAQ_Ai0_timestamp", "red", "green", "FP", 'FPTS']
        for c in channels:
            mCSV = re.match(r"^(?P<GEN>\w{2,3})-(?P<ID>(\d|\w){3,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<DN>[-&\w]+)_(?P<H>([LR]H_)?)(?P<A>p\d+)(?P<SP>[-&\w]*)" +
                            f"_{c}(?P<BSC>(_(BSC)\d)?)" + r"(?P<S>_session\d+_|_?)(?P<T>\d{4}-?\d{2}-?\d{2})T(?P<TD>[_\d]+)\.[(csv)|\d+]", filename)
            if mCSV is not None:
                options = mCSV.groupdict()
                ftype = c
                oS = options["S"]
                options['H'] = ("FP_" + options['H']) if options['H'] else ''
                break
                # print(filename)
                # print(options)
        if ftype is None:
            #print("special:", filename)
            return None
    mS = re.match(r".*(session\d+).*", oS)
    fS = ""
    if mS:
        fS = "_"+mS.group(1)
    options["ftype"] = ftype
    options["animal"] = options['GEN'] + "-" + options["ID"] + "_" + options["EP"]
    options["session"] = options['A'] + fS + (("_"+options['H']) if options['H'] else "")
    return options


def access_mat_with_path(mat, p, ravel=False, dtype=None, raw=False):
    """ Takes in .mat file or hdf5 file and path like structure p and return the entry
    :param mat:
    glml matfiles: modified from Belief_State_FP_Analysis.m legacy Chris Hall GLM structure:
        glml/
            notes/
                hemisphere/
                region/
            time/
                center_in/
                contra/
                contra_rew/
                contra_unrew/
                execute/
                initiate/
                ipsi/
                ipsi_rew/
                ipsi_unrew/
                left_in_choice/
                right_in_choice/
            trial_event_FP_time/
            trials/
                ITI/
                center_in/
                center_to_side/
                contra/
                contra_rew/
                contra_unrew/
                execute/
                initiate/ termination of the trial.
                ipsi/
                ipsi_rew/
                ipsi_unrew/
                left_in_choice/
                omission/
                right_in_choice/
                side_to_center/
                time_indexs/
            value/
                center_to_side_times/
                contra/
                cue_port_side/ 2=left 1=right
                execute/
                initiate/
                ipsi/
                port_side/
                result/ : 1.2=reward, 1.1 = correct omission, 2 = incorrect, 3 = no choice,  0: undefined
                side_to_center_time/
                time_to_left/
                time_to_right/
    :param p:
    :return:
    """
    result = mat
    for ip in p.split("/"):
        result = result[ip]
    if raw:
        return result
    result = np.array(result, dtype=dtype)
    return result.ravel() if ravel else result


def recursive_mat_dict_view(mat, prefix=''):
    """ Recursively print out mat in file structure for visualization, only support pure dataset like"""
    for p in mat:
        print(prefix + p+"/")
        if not isinstance(mat[p], h5py.Dataset) and not isinstance(mat[p], np.ndarray):
            recursive_mat_dict_view(mat[p], prefix+"    ")


###################################################
#################### Cleaning #####################
###################################################
def flip_back_2_channels(animal, session):
    pass



########################################################
#################### Preprocessing #####################
########################################################


def raw_fluor_to_dff(rec_time, rec_sig, iso_time, iso_sig, baseline_method='robust', zscore=False, **kwargs):
    """ Takes in 1d signal and convert to dff (zscore dff)
    :param rec_sig:
    :param rec_time:
    :param iso_sig:
    :param iso_time:
    :param baseline_method:
    :param zscore:
    :param kwargs:
    :return:
    """
    # TODO: figure out the best policy for removal currently no removal
    # TODO: More in-depth analysis of the best baselining approach with quantitative metrics
    bms = baseline_method.split('_')
    fast = False
    if len(bms) > 1:
        fast = bms[-1] == 'fast'
        baseline_method = bms[0]
    if baseline_method == 'robust':
        f0 = f0_filter_sig(rec_time, rec_sig, buffer=not fast, **kwargs)[:, 0]
    elif baseline_method == 'mode':
        f0 = percentile_filter(rec_time, rec_sig, perc=None, **kwargs)
    elif baseline_method.startswith('perc'):
        pc = int(baseline_method[4:])
        f0 = percentile_filter(rec_time, rec_sig, perc=pc, **kwargs)
    elif baseline_method == 'isosbestic':
        # cite jove paper
        reference = interpolate.interp1d(iso_time, iso_sig, fill_value='extrapolate')(rec_time)
        signal = rec_sig
        f0 = get_f0_Martianova_jove(reference, signal)
    elif baseline_method == 'isosbestic_old':
        dc_rec, dc_iso = np.mean(rec_sig), np.mean(iso_sig)
        dm_rec_sig, dm_iso_sig = rec_sig - dc_rec, iso_sig - dc_iso
        # TODO: implement impulse based optimization
        f0_iso = isosbestic_baseline_correct(iso_time, dm_iso_sig, **kwargs) + dc_rec
        f0 = f0_iso
        if iso_time.shape != rec_time.shape or np.allclose(iso_time, rec_time):
            f0 = interpolate.interp1d(iso_time, f0_iso, fill_value='extrapolate')(rec_time)
    else:
        raise NotImplementedError(f"Unknown baseline method {baseline_method}")
    dff = (rec_sig - f0) / (f0 + np.mean(rec_sig)+1e-16) # arbitrary DC shift to avoid issue
    return (dff - np.mean(dff)) / np.std(dff, ddof=1) if zscore else dff


def sources_get_noise_power(s415, s470):
    npower415 = GetSn(s415)
    npower470 = GetSn(s470)
    return npower415, npower470


def get_sample_interval(times):
    return np.around((np.max(times) - np.min(times)) / len(times), 0)


def resample_quasi_uniform(sig, times, method='interpolate'):
    if np.sum(np.diff(times) < 0) > 0:
        shuffles = np.argsort(times)
        sig = sig[shuffles]
        times = times[shuffles]
    si = get_sample_interval(times)
    T0, Tm = np.min(times), np.max(times)
    if method == 'interpolate':
        new_times = np.arange(T0, Tm, si)
        new_sig = interpolate.interp1d(times, sig, fill_value='extrapolate')(new_times)
    elif method == 'fft':
        new_sig, new_times = signal.resample(sig, int((Tm-T0) // si), t=times)
    else:
        raise NotImplementedError(f'unknown method {method}')
    return new_sig, new_times


def denoise_quasi_uniform(sig, times, method='wiener'):
    new_sig, new_times = resample_quasi_uniform(sig, times)
    if method == 'wiener':
        return signal.wiener(new_sig), new_times
    else:
        raise NotImplementedError(f'Unknown method {method}')


def robust_filter(ys, method=12, window=200, optimize_window=2, buffer=False):
    """
    First 2 * windows re-estimate with mode filter
    To avoid edge effects as beginning, it uses mode filter; better solution: specify initial conditions
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=buffer)
    opt_w = int(np.rint(optimize_window * window))
    # prepend
    init_win_ys = ys[:opt_w]
    prepend_ys = init_win_ys[opt_w-1:0:-1]
    ys_pp = np.concatenate([prepend_ys, ys])
    f0 = np.array([(mf(ys_pp, i), mDC.get_dev()) for i in range(len(ys_pp))])[opt_w-1:]
    return f0


def f0_filter_sig(xs, ys, method=12, window=200, optimize_window=2, edge_method='prepend', buffer=False,
                  **kwargs):
    """
    First 2 * windows re-estimate with mode filter
    To avoid edge effects as beginning, it uses mode filter; better solution: specify initial conditions
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=buffer)
    opt_w = int(np.rint(optimize_window * window))
    # prepend
    init_win_ys = ys[:opt_w]
    init_win_xs = xs[:opt_w]
    if edge_method == 'init':
        # subpar method so far, use prepend
        initial = percentile_filter(init_win_xs, init_win_ys, window)
        initial_std = np.sqrt(max(0, np.mean(np.square(init_win_ys - initial))))
        m2 = np.mean(np.square(init_win_ys[init_win_ys - initial < (method % 10) * initial_std]))
        mDC.set_init(np.mean(initial[:window]), np.std(initial, ddof=1))
        dff = np.array([(mf(ys, i), mDC.get_dev()) for i in range(len(ys))])
    elif edge_method == 'prepend':
        prepend_xs = init_win_xs[opt_w-1:0:-1]
        prepend_ys = init_win_ys[opt_w-1:0:-1]
        prepend_xs = 2 * np.min(init_win_xs) - prepend_xs
        ys_pp = np.concatenate([prepend_ys, ys])
        xs_pp = np.concatenate([prepend_xs, xs])
        dff = np.array([(mf(ys_pp, i), mDC.get_dev()) for i in range(len(ys_pp))])[opt_w-1:]
    elif edge_method == 'mode':
        dff = np.array([(mf(ys, i), mDC.get_dev()) for i in range(len(ys))])
        dff[:opt_w, 0] = percentile_filter(init_win_xs, init_win_ys, window)
    else:
        raise NotImplementedError(f"Unknown method {edge_method}")

    return dff


def percentile_filter(xs, ys, window=200, perc=None, **kwargs):
    # TODO: 1D signal only
    if perc is None:
        perc, val = df_percentile(ys[:window])
    return scipy.ndimage.percentile_filter(ys, perc, window)


def isosbestic_baseline_correct_old(xs, ys, window=200, perc=50, **kwargs):
    # TODO: this is the greedy method with only the mean estimation
    #return f0_filter_sig(xs, ys, method=method, window=window)[:, 0]
    return percentile_filter(xs, ys, window, perc)


def isosbestic_baseline_correct(xs, ys, window=200, perc=50, **kwargs):
    # TODO: current use simplest directly import zdff method but want to rigorously test baselining effect
    # TODO: this is the greedy method with only the mean estimation
    #return f0_filter_sig(xs, ys, method=method, window=window)[:, 0]
    return percentile_filter(xs, ys, window, perc)


def calcium_dff(xs, ys, xs0=None, y0=None, method=12, window=200):
    f0 =f0_filter_sig(xs, ys, method=method, window=window)[:, 0]
    return (ys-f0) / f0


def wiener_deconvolution(y, h):
    # TODO: wiener filter performance not well as expected
    # perform wiener deconvolution on 1d array
    T = len(y)
    sn = GetSn(y)
    freq, Yxx = scipy.signal.welch(y, nfft=T)
    Yxx[1:] = Yxx[1:] / 2 # divide evenly between pos and neg
    from scipy.signal import fftconvolve
    Hf = np.fft.rfft(h, n=T)
    Hf2 = Hf.conjugate() * Hf
    Sxx = np.maximum(Yxx - sn**2, 1e-16)
    Nxx = sn ** 2
    Gf = 1 / Hf * (1 / (1 + 1 / (Hf2 * Sxx / Nxx)))
    Yf = np.fft.rfft(y)
    x = np.fft.irfft(Yf * Gf)
    return x, Gf


def inverse_kernel(c, N=None, fft=True):
    """ Computes the deconvolution kernel of c
    :param c:
    :param N:
    :param fft: if True uses fft else uses matrix inversion
    :return:
    """
    if N is None:
        N = len(c) * 2
    if fft:
        cp = np.zeros(N)
        cp[:len(c)] = c
        Hf = np.fft.rfft(cp, n=N)
        return np.fft.irfft(1/Hf)
    else:
        H = spdiags([np.full(N, ic) for ic in c], np.arange(0, -3, step=-1), format='csc')
        G = sp_linalg.inv(H)
        return G[-1, ::-1]


def moving_average(s, window=30, non_overlap=False, pad=False):
    # pad in front
    if non_overlap:
        smoothen = [np.mean(s[i:i + window]) for i in range(0, len(s) - window + 1, window)]
    else:
        smoothen = [np.mean(s[i:i + window]) for i in range(len(s) - window + 1)]
    if pad:
        return np.concatenate((np.full(window-1, smoothen[0]), smoothen))
    else:
        return smoothen


########################################################
###################### Simulation ######################
########################################################
class SpikeCalciumizer:

    MODELS = ['Leogang', 'AR']
    fmodel = "Leogang"
    std_noise = 0.03 # percentage of the saturation level or absolute noise power
    fluorescence_saturation = 0. # 300.
    alpha = 1. #50 uM
    bl = 0
    tauImg = 100  # ms;
    tauCa = 400. #ms
    AR_order = None
    g = None
    ALIGN_TO_FIRST_SPIKE = True
    cutoff = 1000.

    def __init__(self, **params):
        for p in params:
            if hasattr(self, p):
                setattr(self, p, params[p])
            else:
                raise RuntimeError(f'Unknown Parameter: {p}')
        if self.fmodel.startswith('AR'):
            # IndexOutOfBound: not of AR_[order]
            # ValueError: [order] is not int type
            self.AR_order = int(self.fmodel.split('_')[1])
            assert self.g is not None and len(self.g) == self.AR_order
        elif self.fmodel == 'Leogang':
            self.AR_order = 1
            self.g = [1-self.tauImg/self.tauCa]
        else:
            assert self.fmodel in self.MODELS

    # TODO: potentially offset the time signature such that file is aligned with the first spike
    def apply_transform(self, spikes, size=None, sample=None):
        # spikes: pd.DataFrame
        times, neurons = spikes['spike'].values, spikes['neuron'].values
        if self.ALIGN_TO_FIRST_SPIKE:
            times = times - np.min(times) # alignment to 1st spike
        if size is None:
            size = int(np.max(neurons)) + 1
        if sample is None:
            # only keep up to largest multiples of tauImg
            t_end = np.max(times)
        else:
            t_end = sample * self.tauImg
        time_bins = np.arange(0, t_end+1, self.tauImg)
        all_neuron_acts = np.empty((size, len(time_bins) - 1))
        for i in range(size):
            neuron = neurons == i
            all_neuron_acts[i] = np.histogram(times[neuron], time_bins)[0]
        return self.binned_spikes_to_calcium(all_neuron_acts)

    def apply_tranform_from_file(self, *args, sample=None): #TODO: add #neurons to simulated spike,
        # last item possibly
        # args: (index, time) or one single hdf5 file
        if len(args) == 2:
            fneurons, ftimes = args
            assert ftimes[-4:] == '.dat' and fneurons[-4:] == '.dat' \
                   and 'times' in ftimes and 'index' in fneurons
            s_index = np.loadtxt(fneurons, dtype=np.int)
            s_times = np.loadtxt(ftimes, dtype=np.float)
            spikes = pd.DataFrame({'spike': s_times, 'neuron': s_index})
        elif len(args) == 1:
            fspike = args[0]
            assert fspike[-5:] == '.hdf5'
            with h5py.File(fspike, 'r') as hf:
                spikes = pd.DataFrame({'spike': hf['spike'], 'neuron': hf['neuron']})
        else:
            raise RuntimeError("Bad Arguments")
        return self.apply_transform(spikes, sample=sample)

    def binned_spikes_to_calcium(self, neuron_acts, c0=0, fast_inverse=False):
        """
        :param neuron_acts: np.ndarray N x T (neuron x samples)
        :param fast_inverse: whether to use fast reverse. two methods return the same values
        :return:
        """
        # TODO; determine how many spikes were in the first bin
        if len(neuron_acts.shape) == 1:
            print("input must be 2d array with shape (neuron * timestamps)")
        calcium = np.zeros(neuron_acts.shape, dtype=np.float)
        T = neuron_acts.shape[-1]
        fluor_gain = self.alpha * neuron_acts
        if self.AR_order is not None and self.g is not None:
            if fast_inverse:
                G = spdiags([np.ones(T)] + [np.full(T, -ig) for ig in self.g],
                            np.arange(0, -self.AR_order-1, step=-1),format='csc')
                calcium = fluor_gain @ sp_linalg.inv(G.T)
            else:
                calcium[:, 0] = fluor_gain[:, 0]
                for t in range(1, T):
                    ar_sum = np.sum([calcium[:, t-i] * self.g[i-1] for i in range(1, min(t,self.AR_order)+1)],
                                    axis=0)
                    calcium[:, t] = ar_sum + fluor_gain[:, t]
        else:
            raise NotImplementedError(f"Unidentified Model {self.fmodel}")
        if self.fluorescence_saturation > 0:
            calcium = self.fluorescence_saturation * calcium / (calcium + self.fluorescence_saturation)
        calcium += self.bl # TODO: determine whether it is better to add baseline before or after saturation
        if self.std_noise:
            multiplier = self.fluorescence_saturation if self.fluorescence_saturation > 0 else 1
            calcium += np.random.normal(0, self.std_noise * multiplier, calcium.shape)
        return calcium

    def loop_test(self, length, iterations=1000, fast_inv=False):
        # Run time tests of simulation algorithms
        times = [None] * iterations
        N = 10
        for j in range(iterations):
            t0 = time.time()
            rs = np.random.randint(0, 30, (N, length))
            # rs = np.random.random(length)
            self.binned_spikes_to_calcium(rs, fast_inv)
            times[j] = time.time() - t0
        return times


##################################################
################# Visualization ##################
##################################################
def visualize_dist(FP_415_time, FP_415_signal, FP_470_time, FP_470_signal, samples=200):
    s415, s470 = FP_415_signal[:samples], FP_470_signal[:samples]
    dm_470, dm_415 = s470-np.mean(s470), s415 - np.mean(s415)
    print(np.std(dm_470), np.std(dm_415))
    sns.distplot(dm_415, label='415')
    sns.distplot(dm_470, label='470')
    plt.legend()
    #plt.hist([dm_415, dm_470])


def signal_filter_visualize(FP_415_time, FP_415_signal, FP_470_time, FP_470_signal,
                            isosbestic=True, **kwargs):
    # For visualize purpose, all signals are demeaned first:
    # kwargs might not be the best usage here
    # TODO: add exclude event property
    mean_470 = np.mean(FP_470_signal)
    m = mean_470
    mean_415 = np.mean(FP_415_signal)
    FP_470_signal = FP_470_signal - mean_470
    FP_415_signal = FP_415_signal - mean_415
    if isosbestic:
        f0 = isosbestic_baseline_correct(FP_415_time, FP_415_signal+m, **kwargs)
        n415, n470 = sources_get_noise_power(FP_415_signal, FP_470_signal)
        std415, std470 = np.std(FP_415_signal, ddof=1), np.std(FP_470_signal, ddof=1)
        f0_npower_correct = f0 * n470 / n415
        f0_std_correct = f0 * std470 / std415
        bases = {'415_mean': f0, 'f0_npower_correct': f0_npower_correct, 'f0_std_correct': f0_std_correct}
        plt.plot(FP_415_time, FP_415_signal+m, 'm-')
        plt.plot(FP_470_time, FP_470_signal+m, 'b-')
        plt.plot(FP_415_time, np.vstack([f0, f0_npower_correct, f0_std_correct]).T)
        plt.legend(['415 channel (isosbestic)', '470 channel', 'raw baseline', 'noise-power-correct',
                    'sig-power-correct'])
    else:
        f0_rstd = f0_filter_sig(FP_470_time, FP_470_signal+m, **kwargs)[:, 0]
        # similar to Pnevmatikakis 2016 and caiman library
        f0_perc15 = percentile_filter(FP_470_time, FP_470_signal+m, perc=15, **kwargs)
        f0_percAuto = percentile_filter(FP_470_time, FP_470_signal+m, perc=None, **kwargs)
        bases = {'robust': f0_rstd, 'f0_perc15': f0_perc15, 'f0_percAuto': f0_percAuto}
        plt.plot(FP_415_time, FP_415_signal+m, 'm-')
        plt.plot(FP_470_time, FP_470_signal+m, 'b-')
        plt.plot(FP_470_time, np.vstack([f0_perc15, f0_percAuto, f0_rstd]).T)
        plt.legend(['415 channel', '470 channel', '15-percentile', 'mode-percentile', 'robust-std-filter'])
    plt.xlabel('frames')
    plt.ylabel('Fluorescence (demeaned)')
    return bases


def raw_signal_visualize(FP_415_time, FP_415_signal, FP_470_time, FP_470_signal):
    # For visualize purpose, all signals are demeaned first:
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(FP_470_time, FP_470_signal, 'b-')
    axes[0].plot(FP_415_time, FP_415_signal, 'm-')
    axes[0].legend(['470 channel', '415 channel (isosbestic)'])
    axes[0].set_ylabel('Fluorescence')

    FP_470_signal = FP_470_signal - np.mean(FP_470_signal)
    FP_415_signal = FP_415_signal - np.mean(FP_415_signal)
    axes[1].plot(FP_470_time, FP_470_signal, 'b-')
    axes[1].plot(FP_415_time, FP_415_signal, 'm-')
    axes[1].legend(['470 channel', '415 channel (isosbestic)'])
    axes[1].set_xlabel('frames')
    axes[1].set_ylabel('Fluorescence (demeaned)')


def FP_quality_visualization(raw_reference, raw_signal, ftime, initial_time=300, drop_frame=200,
                             time_unit='s', sig_channel='470nm', control_channel='415nm',
                             roi='470nm', roc_method='QDA', tag='', viz=True):
    # Assuming signal has already been properly dropped
    ## TODO: check edge case when there is linalg error leading to auc-roc method error

    ch, control_ch = sig_channel, control_channel
    roi_string = roi.replace(ch, '')
    if roi_string:
        roi_string = roi_string + '_'
    else:
        roi_string = 'ROI_'
    roi_title = roi_string.replace('_', ' ')

    # result_df, pgrid = jove_find_best_param(raw_reference, raw_signal, smooth_win=int(fr), use_raw=False, remove=0)
    # z_reference, z_signal, z_reference_fitted = jove_fit_reference(raw_reference, raw_signal, smooth_win=int(fr),
    #                                                                use_raw=False, remove=0, **pgrid)
    z_reference, z_signal, z_reference_fitted = jove_fit_reference(raw_reference, raw_signal,
                                                                   use_raw=False, remove=0)
    sig_dict = {'reference': z_reference, 'signal': z_signal, 'fitted_ref': z_reference_fitted}
    # selector = z_signal >= (np.median(z_signal) + np.std(z_signal))
    selector = np.abs(z_signal - np.median(z_signal)) >= np.std(z_signal)
    auc_score = auc_roc_2dist(z_reference_fitted[selector], z_signal[selector], roc_method)
    # if not viz:
    #     return None, auc_score, sig_dict
    # selector = np.full(len(z_signal), 1, dtype=bool)
    print(f'Selected {100 * np.sum(selector) / len(selector):.4f}% data')
    # Plot two channels against each other
    fig = None
    if viz:
        fig = plt.figure(figsize=(20, 9))
        gs = GridSpec(nrows=4, ncols=3)
        ax0 = fig.add_subplot(gs[0, :])
        min_time = np.min(ftime)
        segment_sel = ftime <= (min_time + initial_time)

        segment_time = ftime[segment_sel][drop_frame:] - min_time
        normalize = lambda xs: (xs - np.mean(xs)) / np.std(xs)
        sig_segment = normalize(raw_signal[segment_sel][drop_frame:])
        ref_segment = normalize(raw_reference[segment_sel][drop_frame:])
        ax0.plot(segment_time, sig_segment, label=ch)
        ax0.plot(segment_time, ref_segment, label=control_ch)
        ax0.set_ylabel(f'{roi_string}Z(RawF)')
        ax0.set_title(f'{roi_title.title()}Raw {ch} Contrasted With Control (First {initial_time / 60:.2f} Min)')
        ax0.legend()

        ax01 = fig.add_subplot(gs[1, :])
        max_time = np.max(ftime)
        segment_sel1 = ftime >= (max_time - initial_time)
        segment_time1 = ftime[segment_sel1] - min_time
        sig_segment1 = normalize(raw_signal[segment_sel1])
        ref_segment1 = normalize(raw_reference[segment_sel1])
        ax01.plot(segment_time1, sig_segment1, label=ch)
        ax01.plot(segment_time1, ref_segment1, label=control_ch)
        ax01.set_ylabel(f'{roi_string}Z(RawF)')
        ax01.set_title(f'{roi_title.title()}Raw {ch} Contrasted With Control (Last {initial_time / 60:.2f} Min)')
        ax01.legend()

        ax1 = fig.add_subplot(gs[2, :])
        ax1.plot(segment_time, z_signal[segment_sel][drop_frame:], label=f"Z({ch})")
        # ax1.plot(segment_time, z_reference[segment_sel][drop_frame:], label=control_ch)
        ax1.plot(segment_time, z_reference_fitted[segment_sel][drop_frame:], label='~' + control_ch)
        ax1.set_ylabel(f'{roi_string}Z(F)')
        ax1.set_xlabel(f'Rel. Time ({time_unit})')
        ax1.set_title(f'{roi_title.title()}Z {ch} Contrasted With Control (First {initial_time / 60:.2f} Min)')
        ax1.legend()

        # Plot scatter plot visualization of two channels
        ax2 = fig.add_subplot(gs[3, 0])
        ax2.plot(z_reference[selector], z_signal[selector], 'b.')
        ax2.plot(z_reference, z_reference_fitted, 'r--', linewidth=1.5)
        ax2.set_xlabel(f'{control_ch} values')
        ax2.set_ylabel(f'{ch} values')
        ax3 = fig.add_subplot(gs[3, 1])

        sns.histplot(z_reference_fitted[selector], label=control_ch, kde=False, ax=ax3, color='b')
        # sns.histplot(z_reference_fitted[selector], label=control_ch, kde=True, ax=ax1, color='b')
        sns.histplot(z_signal[selector], label=ch, kde=False, ax=ax3, color='r')
        ax3.legend()
        sns.despine()

        ax4 = fig.add_subplot(gs[3, 2])
        sns.histplot(z_signal[selector] - z_reference_fitted[selector], kde=True, ax=ax4)
        ax4.legend(['diff(470, ~415)'])
        sns.despine()
        if tag:
            tag = tag + ' '
        plt.subplots_adjust(hspace=0.4)
        fig.suptitle(f'{tag}{ch} {roi_title}auc-roc score ({roc_method}): {auc_score:.4f}', fontsize='xx-large')

    return fig, auc_score, sig_dict


def FP_viz_whole_session(raw_reference, raw_signal, ftime, interval=600, drop_frame=200,
                         time_unit='s', sig_channel='470nm', control_channel='415nm',
                         roi='470nm', tag=''):
    # Assuming signal has already been properly dropped
    ## TODO: check edge case when there is linalg error leading to auc-roc method error
    smooth_win, lambd, porder, itermax = 10, 5e4, 1, 50
    smoothened_reference = smooth_signal(raw_reference, smooth_win)
    smoothened_signal = smooth_signal(raw_signal, smooth_win)
    # Find the baseline
    r_base = airPLS(smoothened_reference.T, lambda_=lambd, porder=porder, itermax=itermax)
    s_base = airPLS(smoothened_signal.T, lambda_=lambd, porder=porder, itermax=itermax)

    ch, control_ch = sig_channel, control_channel
    roi_string = roi.replace(ch, '')
    if roi_string:
        roi_string = roi_string + '_'
    else:
        roi_string = 'ROI_'
    roi_title = roi_string.replace('_', ' ')

    palettes = {'sig': sns.color_palette('icefire')[:2],
                'ref': sns.color_palette('icefire')[-2:][::-1]}

    min_time, max_time = np.min(ftime), np.max(ftime)
    nseq = int(np.ceil((max_time - min_time) / interval))
    nrow = int(np.ceil(nseq / 3))
    all_lw, ref_off = 0.8, 2.5
    fig, axes = plt.subplots(nrows=nrow, ncols=3, figsize=(21, 2 * nrow))

    for i in range(nseq):
        start = min_time + i * interval
        end = min_time + (i + 1) * interval
        segment_sel = (ftime <= end) & (ftime >= start)
        segment_time = ftime[segment_sel][drop_frame:] - min_time
        normalize = lambda xs: (xs - np.mean(xs)) / np.std(xs)
        normalize_wms = lambda xs, m, s: (xs - m) / s
        # sig_segment = normalize(raw_signal[segment_sel][drop_frame:])
        sig_segment = raw_signal[segment_sel][drop_frame:]
        ssm, sss = np.mean(sig_segment), np.std(sig_segment)
        sig_segment = (sig_segment - ssm) / sss
        ref_segment = raw_reference[segment_sel][drop_frame:]
        rsm, rss = np.mean(ref_segment), np.std(ref_segment)
        ref_segment = (ref_segment - rsm) / rss
        sb_segment = (s_base[segment_sel][drop_frame:] - ssm) / sss
        rb_segment = (r_base[segment_sel][drop_frame:] - rsm) / rss

        axes.ravel()[i].plot(segment_time, sig_segment, label=ch, lw=all_lw, color=palettes['sig'][0])
        axes.ravel()[i].plot(segment_time, ref_segment - ref_off, label=control_ch, lw=all_lw, color=palettes['ref'][0])
        axes.ravel()[i].plot(segment_time, sb_segment, label=ch + '_trend', lw=all_lw, color=palettes['sig'][1])
        axes.ravel()[i].plot(segment_time, rb_segment - ref_off, label=control_ch + '_trend', lw=all_lw,
                             color=palettes['ref'][1])
        axes.ravel()[i].set_ylabel(f'{roi_string}Z(RawF)')
        axes.ravel()[i].set_xlabel(f'Rel. Time ({time_unit})')
        axes.ravel()[i].set_xlim([start - min_time - 0.05 * interval, end - min_time + 0.05 * interval])
        pend = 'th'
        if i == 0:
            pend = 'st'
        elif i == 1:
            pend = 'nd'
        axes.ravel()[i].set_title(
            f'{roi_title.title()}Raw {ch} Contrasted With Control ({i + 1}{pend} {interval / 60:.2f} Min)')
        if i == 0:
            axes.ravel()[i].legend()

    sns.despine()

    if tag:
        tag = tag + ' '
    plt.subplots_adjust(hspace=0.7)
    fig.suptitle(f'{tag}{ch} {roi_title} raw signal visualization', fontsize='xx-large')
    return fig


#############################################################
#################### Process Management #####################
#############################################################


class ProgressBar:

    """
    Prints remaining time of the process

    Example:
    --------
    >>> N_task = 3
    >>> pbar = ProgressBar(N_task)
    >>> for i in range(N_task):
    ...     pbar.loop_start()
    ...     time.sleep(1)
    ...     pbar.loop_end(i)
    prints:
    Done with 0, estimated run time left: 0h:0m:2.0s
    Done with 1, estimated run time left: 0h:0m:1.0s
    Done with 2, estimated run time left: 0h:0m:0.0s
    TODO: implement more detailed p
    rogress with subtasks
    TODO: implement ability to resume interrupted processes
    """

    def __init__(self, total_sessions):
        self.N = total_sessions
        self.start = None
        self.avgtime = 0
        self.numberDone = 0

    def tstr(self, t):
        return f"{int(t // 3600)}h:{int(t % 3600 // 60)}m:{t % 60:.1f}s"

    def loop_start(self):
        if self.start is None:
            print(f'Starting {self.N} tasks...')
            self.start = time.time()

    def loop_end(self, task_name):
        run_time = time.time() - self.start
        self.numberDone += 1
        self.avgtime = run_time / self.numberDone
        ETA = self.avgtime * (self.N - self.numberDone)
        print(f'Done with {task_name}, estimated run time left: {self.tstr(ETA)}')
        if ETA == 0.:
            print(f'Finished all {self.N} tasks. Total Run Time: {self.tstr(time.time()-self.start)}.')

    def loop_skip(self, task_name):
        self.N -= 1
        assert self.N >= 0
        # run_time = time.time() - self.start
        # self.avgtime = run_time / self.numberDone
        ETA = self.avgtime * (self.N - self.numberDone)
        print(f'Skipping {task_name}, estimated run time left: {self.tstr(ETA)}')
        if ETA == 0.:
            print(f'Finished all {self.N} tasks. Total Run Time: {self.tstr(time.time()-self.start)}.')


########################################################
#################### Miscellaneous #####################
########################################################
def df_col_is_str(df, c):
    return df[c].dtype == object and isinstance(df.iloc[0][c], str)


def df_select_kwargs(df, return_index=False, **kwargs):
    """ R style select for pd.DataFrame
    Alternatively can use query method but very ugly syntax
    def query_method(pse, proj, qarg=""):
        self=pse
        proj_sel = self.meta['animal'].str.startswith(proj)
        full_qarg = f"animal.str.startswith('{proj}').values"
        if qarg:
            full_qarg = full_qarg + ' & ' + qarg
        print(full_qarg)
        return self.meta.query(full_qarg)
    query_method(pse, 'BSD', f"(animal_ID=='{animal}') & (session=='{session}')")
    """
    for kw in kwargs:
        if kw in df.columns:
            karg = kwargs[kw]
            if not hasattr(karg, '__call__'):
                kwargs[kw] = (lambda a: (lambda s: s == a))(karg)
        else:
            logging.warning(f'keyword argument key {kw} is not in dataframe!')
    # add function capacity
    df_sel = np.logical_and.reduce([kwargs[kw](df[kw]) for kw in kwargs if kw in df.columns])
    if return_index:
        return df_sel
    else:
        return df[df_sel]


def df_melt_lagged_features(df, feat, id_vars, value_vars=None):
    if value_vars is None:
        value_vars = [c for c in df.columns if (feat in c)]
    df = pd.melt(df, id_vars, value_vars, f'{feat}_arg', value_name=f'{feat}_value')
    df[f'{feat}_lag'] = df[f'{feat}_arg'].str.replace(feat, '').apply(lambda x: x[1:-1].replace('t', ''))
    df.loc[df[f'{feat}_lag'] == '', f'{feat}_lag'] = 0
    df[f'{feat}_lag'] = df[f'{feat}_lag'].astype(np.int)
    df.drop(columns=f'{feat}_arg', inplace=True)
    return df


def pds_is_valid(pds):
    if pd.api.types.is_string_dtype(pds):
        return ~(pds.isnull() | (pds == ''))
    else:
        return ~pds.isnull()


def pds_neq(x, y):
    return pds_is_valid(x) & pds_is_valid(y) & (x != y)


def decode_from_regfeature(feature):
    get_lag = lambda s: int(s.split('_')[1][:-4])
    if ':' in feature:
        ftype = 'C:R'
        lag = get_lag(feature.split(':')[0])
    elif feature == 'Intercept':
        ftype = feature
        lag = 0
    else:
        ftype = feature.split('_')[0]
        lag = get_lag(feature)
    return ftype, lag


def draw_class_tree(cls_obj):
    # from cogmodels_base import *
    # dot = draw_class_tree(CogModel)
    #
    # dot.render('class_tree', view=True)
    dot = graphviz.Digraph()

    def add_nodes(c):
        dot.node(c.__name__)
        for sub_cls in c.__subclasses__():
            dot.edge(c.__name__, sub_cls.__name__)
            add_nodes(sub_cls)

    add_nodes(cls_obj)
    return dot

