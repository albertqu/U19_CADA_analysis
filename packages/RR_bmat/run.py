# author: Lexi Zhou

from mainAnalysis import *
import os
from clean_bonsai_output import *
from eventcodedict import *

input_folder = '/Users/lexizhou/Desktop/RRM040'


def list_files(dir, type):
    """
    List all files of a certain type in the given dir
    :param dir: directory
    :param type: str
    :return:
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(type) and 'trial' not in name and 'bonsai' not in name:
                r.append(os.path.join(root, name))
    return r


def write_trials(input_folder):
    """
    take in bonsai log files and output trial csv
    :param input_folder: folder that contains bonsai log files, doesn't need to be homogenous
    :return:
    """
    bonsai_files = list_files(input_folder, '.csv')
    for file in bonsai_files:
        pathPrefix = os.path.dirname(file)
        sessionname = str(file).split('/')[-1].split('.')[0].split('2021')[0]
        if not os.path.isdir(pathPrefix + "/" + sessionname):
            os.mkdir(pathPrefix + "/" + sessionname)

        # Save raw bonsai output with event description --> raw behavior LOG human readable
        events = preprocessing(file, eventcodedict_full)
        list_of_bonsaievents = write_bonsaiEvent_dll(events)
        bonsaiEvent_df = write_dll_to_df(list_of_bonsaievents)
        bonsaiEvent_df.to_csv(pathPrefix + "/" + sessionname + "/" + "raw_bonsai_" + sessionname + '.csv')

        # Save selected bonsai events --> cleaned behavior LOG, dropping nonsense
        events_partial = detect_keyword_in_event(preprocessing(file, eventcodedict_partial))
        events_list_partial = clean_and_organize(events_partial)
        list_of_bonsaievents_partial = write_bonsaiEvent_dll(events_list_partial)
        bonsaiEventPartial_df = write_dll_to_df(list_of_bonsaievents_partial)
        bonsaiEventPartial_df.to_csv(pathPrefix + "/" + sessionname + "/" + "bonsai_" + sessionname + '.csv')
        
        # trial structure containing pseudotrials
        trials = trial_writer(list_of_bonsaievents_partial)
        trial_info_filler(trials)
        trial_merger(trials)
        write_lap_block(trials)
        add_stimulation_events(trials, events)

        trials_df = write_trial_to_df(trials)
        trials_df.to_csv(pathPrefix + "/" + sessionname + "/" + "all_trials_" + sessionname + '.csv')
        
        # trial structure with only valid trials
        valid_trials_df = save_valid_trial(trials_df)
        valid_trials_df.to_csv(pathPrefix + "/" + sessionname + "/" + "trials_" + sessionname + '.csv')


write_trials(input_folder)

