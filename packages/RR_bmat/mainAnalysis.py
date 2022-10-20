# author: Lexi Zhou


import pandas as pd
import list as ls
import re
import numpy as np
from copy import deepcopy
import classes as cl


def preprocessing(filepath, eventcodedict):
    names = ['timestamp', 'eventcode']
    bonsai_output = pd.read_csv(filepath, sep=" ", index_col=False, names=names)[['timestamp', 'eventcode']]

    def strip(t):
        if type(t) is str:
            t = t.replace(" ", "")
        return t

    keys = list(eventcodedict.keys())
    bonsai_output['timestamp'] = bonsai_output['timestamp'].map(strip).astype(float)
    bonsai_output = bonsai_output[bonsai_output.eventcode.isin(keys)].reset_index(drop=True)
    bonsai_output['event'] = bonsai_output['eventcode'].map(lambda code: eventcodedict[code])
    first_timestamp = bonsai_output.iloc[0, 0]
    first = bonsai_output[bonsai_output['eventcode'] == 9].index[0]
    bonsai_output = bonsai_output[bonsai_output['eventcode'] != 9].reset_index(drop=True)
    bonsai_output['timestamp'] = bonsai_output['timestamp'].map(lambda t: (t - first_timestamp) / 1000)
    bonsai_output = bonsai_output[['event', 'timestamp', 'eventcode']]
    bonsai_output_final = bonsai_output[first:]

    events_list = bonsai_output_final.values.tolist()

    def restaurant_extractor(events_list):
        for i in range(len(events_list)):
            if int(events_list[i][-1]) == 199:
                # back up to offer tone
                for j in range(i-1, -1, -1):
                    if '% offer' in events_list[j][0]:
                        offer_index = j
                        break
                restaurant = re.findall('R[1-4]+', events_list[offer_index][0])[0]
                events_list[i].append(int(restaurant[1:]))
            if int(events_list[i][-1]) == 99:
                # forward to offer tone
                for j in range(i+1, len(events_list)):
                    if '% offer' in events_list[j][0]:
                        offer_index = j
                        break
                restaurant = re.findall('R[1-4]+', events_list[offer_index][0])[0]
                events_list[i].append(int(restaurant[1:]))
            if len(events_list[i]) <= 3:
                restaurant = re.findall('R[1-4]+', events_list[i][0])[0]
                events_list[i].append(int(restaurant[1:]))

    restaurant_extractor(events_list)

    return events_list


def detect_keyword_in_event(events):
    """
    events -- List: list of bonsai events
    """

    def detect_keyword(string):
        keyword = None
        if string.__contains__('hall'):
            keyword = 'hall'
        elif string.__contains__('zone'):
            keyword = 'offer zone'
        elif string.__contains__('offer'):
            probability = re.findall('[0-9]+', string)[1]
            keyword = probability + '_offer'
        elif string.__contains__('SHARP Accept') or string.__contains__('Enter'):
            keyword = 'enter'
        elif string.__contains__('Quit'):
            keyword = 'quit'
        elif string.__contains__('no-reward'):
            keyword = 'noreward'
        elif string.__contains__('taken'):
            keyword = 'taken'
        elif string.__contains__('reward tone'):
            keyword = 'rewarded'
        elif string.__contains__('Reject'):
            keyword = 'reject'
        elif string.__contains__('Entry'):
            keyword = 'tentry'
        elif string.__contains__('Servo'):
            keyword = 'servo open'
        return keyword

    new_events_list = []
    for i in events:
        if len(i) < 5:
            i_description = i[0]
            i += [detect_keyword(i_description)]
            new_events_list.append(i)
        else:
            new_events_list.append(i)

    return [lis for lis in new_events_list if len(lis) >= 5]


def write_bonsaiEvent_dll(events_list):
    list_of_bonsaievents = ls.DoublyLinkedList()
    for i in events_list:
        event_object = cl.BonsaiEvent(i)
        list_of_bonsaievents.add_to_end(event_object)
    return list_of_bonsaievents


def write_dll_to_df(dll):
    df = pd.DataFrame.from_dict([dll.sentinel.next.info()])
    current = dll.sentinel.next.next
    while current != dll.sentinel:
        current_df = pd.DataFrame.from_dict([current.info()])
        df = pd.concat([df, current_df], axis=0)
        current = current.next
    df = df.drop(columns=['item', 'next','prev'])
    return df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Represent Trials as a DLL of bonsai_event objects in the same restaurant"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_bonsai_event_item(item):
    """Create a new bonsai event object with next and prev = None"""
    new_object = cl.BonsaiEvent(item)
    return new_object


def trial_writer(events_list):
    current = events_list.sentinel.next
    i = 0
    trial = cl.Trial(get_bonsai_event_item(current.item), [current.item], i)
    i+=1
    current_restaurant = current.restaurant

    current = current.next
    trials = ls.DoublyLinkedList()

    while current != events_list.sentinel:
        if current.restaurant == current_restaurant:
            """mouse in the same restaurant"""
            trial.item.append(current.item)
        elif current.restaurant != current_restaurant:
            """mouse in a new restaurant"""
            trials.add_to_end(trial)
            trial = cl.Trial(get_bonsai_event_item(current.item), [current.item], i)
            trial.item.append(current.item)
            i += 1
        current_restaurant = current.restaurant
        current = current.next
    if i == 0:
        trials.add_to_end(trial)
    return trials


def getindex(lists, value):
    return next(i for i, v in enumerate(lists) if value in v)


def trial_info_filler(trials):
    """
    Fill in information about each trial by interating through all trials
    :param trials: DLL of Trial Objects
    :return: Modifies trials, returns nothing
    """
    current_trial = trials.sentinel.next
    while current_trial != trials.sentinel:  # current_trial is a Trial object
        """
        current_trial: Trial Object
        current_trial.sentinel.next.item: DLL of bonsai events
        current_trial.sentinel.next.item.sentinel.next: a single bonsai event with next and prev
        current_trial.sentinel.next.item.sentinel.next.restaurant: restaurant of that single bonsai event
        """

        """Fill Restaurant"""
        current_trial.restaurant = current_trial.item[0][-2]
        current_trial.enter = current_trial.item[0][1]
        current_trial.exit = current_trial.item[-1][1]

        def trial_detector(sublist, index):
            if 'enter' in str(sublist[index:-1]): # Enter can't be the last event
                """accepting offer"""
                enter_index = getindex(sublist, 'enter')
                current_trial.choice = sublist[enter_index][1]
                current_trial.accept = 1
                if 'servo open' in str(sublist[index:]):
                    """reward presented"""
                    servo_index = getindex(sublist[index:], 'servo open') + index
                    current_trial.reward = 1
                    current_trial.outcome = sublist[servo_index][1]
                    if 'taken' in str(sublist[servo_index:]):
                        """reward collected, trial ends"""
                        taken_index = getindex(sublist[servo_index:], 'taken') + servo_index
                        current_trial.collection = sublist[taken_index][1]
                        current_trial.trial_end = sublist[taken_index][1]
                        current_trial.comment = 'accepted, waited, rewarded, and collected'
                    elif 'quit' in str(sublist[servo_index:]):
                        """exit wait zone after servo open without taking pellets"""
                        quit_index = getindex(sublist[servo_index:], 'quit') + servo_index
                        current_trial.trial_end = sublist[quit_index][1]
                        current_trial.comment = 'rewarded but pellet not taken'
                elif 'taken' in str(sublist[enter_index:]):
                    """servo open missing but pellet taken"""
                    taken_index = getindex(sublist[enter_index:], 'taken') + enter_index
                    current_trial.reward = 1
                    current_trial.collection = sublist[taken_index][1]
                    current_trial.trial_end = sublist[taken_index][1]
                    current_trial.comment = 'servo open timestamp missing but pellet taken'
                elif 'noreward' in str(sublist[enter_index:]):
                    noreward_index = getindex(sublist[enter_index:], 'noreward') + enter_index
                    current_trial.outcome = sublist[noreward_index][1]
                    current_trial.trial_end = sublist[noreward_index][1]
                    current_trial.comment = 'accepted, waited, no-reward'
                elif 'quit' in str(sublist[enter_index:]):
                    quit_index = getindex(sublist[enter_index:], 'quit') + enter_index
                    current_trial.quit = sublist[quit_index][1]
                    current_trial.trial_end = sublist[quit_index][1]
                    current_trial.comment = 'animal accepted but quit before outcome revealed'
                elif 'reject' in str(sublist[enter_index:]):
                    reject_index = getindex(sublist[enter_index:], 'reject') + enter_index
                    current_trial.quit = sublist[reject_index][1]
                    current_trial.trial_end = sublist[reject_index][1]
                    current_trial.comment = 'animal accepted but quit but quit wasnot timestamped'
                else:
                    current_trial.quit = sublist[-1][1]
                    current_trial.trial_end = sublist[-1][1]
                    current_trial.comment = 'animal accepted but quit'
            elif 'reject' in str(sublist[index:]):
                """existed t junction and went to the next restaurant"""
                reject_index = getindex(sublist[index:], 'reject') + index
                current_trial.trial_end = sublist[reject_index][1]
                current_trial.choice = sublist[reject_index][1]
                current_trial.comment = 'entered T junction then rejected and moved on'
            else:
                current_trial.choice = sublist[-1][1]
                current_trial.trial_end = sublist[-1][1]
                current_trial.comment = 'entered T junction but never entered restaurant'

        """Detect Offer"""
        # Create a deep copy such that the original object variable won't be modified
        event_track = deepcopy(current_trial.item)

        """Write events"""
        for j in range(len(event_track)):
            if "_offer" in str(event_track[j]):
                current_trial.tone_onset = event_track[j][1]
                current_trial.tone_prob = event_track[j][-1].split('_')[0]
                if 'tentry' in str(event_track[j:]):
                    """check if tentry happened"""
                    tentry_index = getindex(event_track, 'tentry')
                    current_trial.T_Entry = event_track[tentry_index][1]
                    trial_detector(event_track, tentry_index)
                elif 'enter' in str(event_track[j:]):
                    """t junction entry somehow was not captured"""
                    current_trial.comment = 't junction entry timestamp missing'
                    trial_detector(event_track, j)
                elif 'reject' in str(event_track[j:]):
                    """rejecting offer and backtrack into the hallway in current restaurant"""
                    reject_index = getindex(event_track, 'reject')
                    current_trial.choice = event_track[reject_index][1]
                    current_trial.trial_end = event_track[reject_index][1]
                    current_trial.comment = 'rejected offer but t junction entry not detected'
            if "_offer" not in str(event_track) and "taken" in str(event_track[j]):
                """offer tone missing but reward was taken"""
                current_trial.reward = 1
                current_trial.accept = 1
                current_trial.collection= event_track[j][1]
                current_trial.trial_end = event_track[j][1]
                current_trial.comment = 'offer tone timestamp missing, pellet taken'
                if 'enter' in str(event_track[:j]):
                    enter_index = getindex(event_track[:j], 'enter')
                    current_trial.choice = event_track[enter_index][1]
                    current_trial.comment += ', choice = restaurant entry timestamp'
                if 'tentry' in str(event_track[:j]):
                    tentry_index = getindex(event_track[:j], 'tentry')
                    current_trial.T_Entry = event_track[tentry_index][1]
                if 'servo open' in str(event_track[:j]):
                    servo_index = getindex(event_track[:j], 'servo open')
                    current_trial.outcome = event_track[servo_index][1]
            if "_offer" not in str(event_track) and 'noreward' in str(event_track[j]):
                """offer tone missing but no-reward timestampped"""
                current_trial.outcome = event_track[j][1]
                current_trial.trial_end = event_track[j][1]
                current_trial.comment = 'offer tone timestamp missing but noreward timestampped'
                if 'enter' in str(event_track[:j]):
                    enter_index = getindex(event_track[:j], 'enter')
                    current_trial.choice = event_track[enter_index][1]
                    current_trial.comment += ', choice = restaurant entry timestamp'
                if 'tentry' in str(event_track[:j]):
                    tentry_index = getindex(event_track[:j], 'tentry')
                    current_trial.T_Entry = event_track[tentry_index][1]
        current_trial = current_trial.next


def trial_merger(trials):
    current_trial = trials.sentinel.next
    while current_trial != trials.sentinel:
        if 'offer tone timestamp missing' in str(current_trial.comment):
            restaurant = current_trial.restaurant
            check = current_trial.prev
            steplimit = 5
            for i in range(steplimit):
                if check != trials.sentinel and current_trial.next != trials.sentinel:
                    if check.tone_onset is not None and check.restaurant == restaurant:
                        current_trial.tone_onset = check.tone_onset
                        current_trial.tone_prob = check.tone_prob
                        current_trial.enter = check.enter
                        current_trial.comment = 'fetched offer tone'
                        if current_trial.outcome is None:
                            current_trial.outcome = check.outcome
                        if current_trial.choice is None:
                            current_trial.choice = check.choice
                        if current_trial.T_Entry is None:
                            current_trial.T_Entry = check.T_Entry
                        check.prev.next = check.next
                        check.next.prev = check.prev
                        break
                i += 1
                check = check.prev
        current_trial = current_trial.next


def add_stimulation_events_old(trials, eventslist):
    current_trial = trials.sentinel.next
    events = np.array(eventslist)
    while current_trial != trials.sentinel:
        if current_trial.tone_onset:
            start = current_trial.tone_onset
            end = current_trial.exit
            stim_index = np.where((events[:, 1].astype(float) > start) & (events[:, 1].astype(float) < end))
            for i in stim_index[0]:
                if 99 in events[i-5:i, 2].astype(int) and current_trial.stimulation_on is None:
                    on_index = np.where(events[i-5:i, 2].astype(int) == 99)[0]
                    if events[i-5+on_index, 3][0] == events[i, 3]:
                        current_trial.stimulation_on = events[i-5+on_index, 1][0]
                if 199 in events[i:i+5, 2].astype(int):
                    if current_trial.stimulation_off is None and current_trial.stimulation_on:
                        off_index = np.where(events[i:i+5, 2].astype(int) == 199)[0]
                        if events[i+off_index, 3][0] == events[i, 3] and \
                                (events[i+off_index, 1][0].astype(float) - float(current_trial.stimulation_on)) <= 2:
                            current_trial.stimulation_off = events[i+off_index, 1][0]
            if current_trial.stimulation_off is None and current_trial.stimulation_on:
                current_trial.stimulation_on = None
        current_trial = current_trial.next


def resort_trial_DLL(trials):
    # Function to sort trial doubly linked list so that tone_onsets are sorted from start to end
    cursor = trials.sentinel.next
    prev = cursor.tone_onset
    while prev is None:
        cursor = cursor.next
        prev = cursor.tone_onset

    def float_up(trials, node):
        # float up if there is NaN above or nodes with LATER tone_onset
        curr = node.prev
        while curr != trials.sentinel:
            if curr.tone_onset is None:
                trials.swap_nodes(curr, node)
                curr = node.prev
            else:
                if node.tone_onset < curr.tone_onset:
                    # try:
                    trials.swap_nodes(curr, node)
                    # except:
                    #     print(curr.tone_onset, node.tone_onset)
                    curr = node.prev
                else:
                    break
    freeze = False
    while cursor.next != trials.sentinel:
        curr = cursor.next.tone_onset
        if curr is not None:
            if curr < prev:
                float_up(trials, cursor.next)
                freeze = True
            else:
                prev = curr
        if freeze:
            freeze = False
        else:
            cursor = cursor.next


def add_stimulation_events(trials, eventslist):
    """ Takes in DLL trials and add stimulation features when applicable in trial_node.stimulation_on/off
    trials: DLL with trial structure
    eventslist: list of event triples (ename, time, code)
    """
    START = eventslist[0][1]

    def prev_trial_tone_onset(ctrial):
        curr = ctrial.prev
        while curr != trials.sentinel:
            if curr.tone_onset:
                return curr.tone_onset
        return START

    current_trial = trials.sentinel.next

    def find_in_elist(start, elist, t):
        for i in range(start, len(elist)):
            if elist[i][1] == t:
                return i
    curr_eind = 0
    while current_trial != trials.sentinel:
        if current_trial.tone_onset:
            # iterate on events
            # find corresponding event in elist
            prev_eind = curr_eind
            curr_eind = find_in_elist(curr_eind, eventslist, current_trial.tone_onset)
            # look backwards in time to find stimOn
            for j in range(curr_eind-1, prev_eind - 1, -1):
                if eventslist[j][2] == 99:
                    current_trial.stimulation_on = eventslist[j][1]
            # look forward in time to find stimOff
            if current_trial.stimulation_on is not None:
                for j in range(curr_eind + 1, len(eventslist)):
                    if eventslist[j][2] == 199:
                        current_trial.stimulation_off = eventslist[j][1]
        current_trial = current_trial.next


def write_lap_block(trials):

    sequence = {
        1: 2,
        2: 3,
        3: 4,
        4: 1
    }

    current_trial = trials.sentinel.next
    if current_trial.trial_end:
        current_trial.lapIndex, current_trial.blockIndex = 0, 0
    block = 0
    lap = 0
    current_trial = current_trial.next
    while current_trial != trials.sentinel.prev:
        if current_trial.trial_end:
            if current_trial.prev.trial_end is None and current_trial.prev.tone_onset is None:
                block += 1
                lap = 0
            elif sequence[current_trial.prev.restaurant] == current_trial.restaurant:
                if current_trial.prev.restaurant == 4:
                    lap += 1
            current_trial.lapIndex = lap
            current_trial.blockIndex = block
        elif current_trial.trial_end is None and current_trial.tone_onset is None:
            if current_trial.prev.trial_end and current_trial.prev.tone_onset:
                block += 1
                lap = 0
            current_trial.blockIndex = block
            current_trial.lapIndex = lap
        current_trial = current_trial.next


def write_trial_to_df(trials):
    """
    trials -- DLL: DLL representation of trials
    return -- dataFrame
    """
    current = trials.sentinel.next
    df = pd.DataFrame.from_dict([trials.sentinel.next.info()])
    while current != trials.sentinel:
        current_df = pd.DataFrame.from_dict([current.info()])
        df = pd.concat([df, current_df])
        current = current.next
    df = df.drop(columns=['item', 'firstEventNode', 'next', 'prev', 'enter']).iloc[1:, :]
    df = df.rename(columns={'index': 'trial_index'}).set_index('trial_index')
    return df


def save_valid_trial(df):
    df = df[df.loc[:, ['trial_end', 'tone_onset', 'T_Entry']].notnull().all(axis='columns')]
    df = df.query('tone_onset < T_Entry and T_Entry < choice and trial_end <= exit')

    new_df = df.sort_values(by=['tone_onset'])
    return new_df
