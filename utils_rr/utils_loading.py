import pandas as pd
import numpy as np
import os
from os.path import join as oj
from utils_rr.poseseries import PoseSeries

def load_data_sessions_gcamp():
    sessions = {'RRM026': {151: 2, 160: 2, 167: 2, 172: 2},
            'RRM027': {155: 1, 170: 2, 175: 3},
            'RRM028': {123: 2, 130: 2, 136: 1, 141: 3, 151: 3, 156: 3},
            'RRM029': {125: 2, 130: 2, 141: 2, 153: 2, 158: 3},
            'RRM030': {139: 3, 143: 3, 146: 3, 149: 2, 154: 3, 159: 3},
            'RRM031': {125: 2, 130: 1, 134: 3, 139: 1, 143: 3, 146: 3, 149: 2},
            'RRM032': {118: 1, 122: 3, 128: 3, 132: 1, 135: 3, 138: 2, 143: 3,
                       147: 3},
            'RRM033': {118: 1, 122: 2, 132: 2, 135: 2, 138: 2, 143: 3, 147: 3},
            'RRM035': {195: 1, 198: 1},
            'RRM036': {161: 1, 169: 1, 172: 1, 176: 3}}
    return sessions

def load_neural_data(rse, cache_folder, sessions, method='ZdF_jove'):
    animal_list_parquet = oj(cache_folder, f'RRM26-36_nb_df_{method}.pq')
    rse.nbm.event_time_windows = {'tone_onset': np.arange(-1, 1.001, 0.05),
                                'T_Entry': np.arange(-1, 1.001, 0.05),
                                'choice': np.arange(-1, 1.001, 0.05),
                                'outcome': np.arange(-1, 1.001, 0.05),
                                'quit': np.arange(-1, 1.001, 0.05),
                                'collection': np.arange(-0.5, 1.001, 0.05),
                                'trial_end': np.arange(-1, 1.001, 0.05),
                                'exit': np.arange(-1, 1.001, 0.05)}
    neur_events = ['T_Entry', 'tone_onset', 'quit', 'outcome', 'collection']

    if not os.path.exists(animal_list_parquet):
        nbdfs = []
        for animal in sessions:
            print(animal)
            animal_sessions = sessions[animal]
            for s, hemi in animal_sessions.items():
                print(s, hemi)
                nb_df_i = rse.align_lagged_view('RRM', neur_events, method=method, #'lossless',
                                        animal=animal, session=f'Day{s}')
                if nb_df_i is not None:
                    if hemi == 1: 
                        nb_df_i = nb_df_i[nb_df_i['roi'] == 'right_470nm'].reset_index(drop=True)
                    elif hemi == 2:
                        nb_df_i = nb_df_i[nb_df_i['roi'] == 'left_470nm'].reset_index(drop=True) 
                    nbdfs.append(nb_df_i)

        nb_df = pd.concat(nbdfs, axis=0)
        nb_df = nb_df.reset_index(drop=True)
        nb_df['hall_time'] = nb_df['T_Entry'] - nb_df['tone_onset']
        nb_df['decision_time'] = nb_df['choice'] - nb_df['tone_onset']
        nb_df['decision'] = nb_df['accept'].map({0: 'reject', 1: 'accept'})
        nb_df.loc[~nb_df['quit'].isnull(), 'decision'] = 'quit'
        nb_df['trial_type'] = nb_df['decision']
        nb_df.loc[nb_df['decision'] == 'accept', 'trial_type'] = 'unrewarded'
        nb_df.loc[(nb_df['decision'] == 'accept') & (nb_df['reward'] == 1), 'trial_type'] = 'rewarded'
        nb_df.to_parquet(animal_list_parquet)
    else:
        nb_df = rse.align_lagged_view_parquet(animal_list_parquet)
        nb_df = nb_df.reset_index(drop=True)

    nb_df['offer'] = nb_df['offer_prob'].map({0.0: r'0%+7s', 20.0: r'20%+5s', 80.0: r'80%+3s', 100.0: r'100%+1s'})
    # nb_df.rename(columns={'tone_prob': 'offer_prob'}, inplace=True)
    nb_df = add_past_outcome_features(nb_df, dt=120, inplace=False)
    # nb_df['hallt_bin'] = pd.cut(nb_df['hall_time'], [0, 0.37,  0.44,  0.6, 1, 2])
    ht_bin4 = [0, 0.45, 0.54, 0.73, 2]
    nb_df['hallt_4bin'] = pd.cut(nb_df['hall_time'], ht_bin4)
    nb_df['hallt_4bin_w'] = pd.cut(nb_df['hall_time'], ht_bin4, labels=['fast', 'mid', 'slow', 'tail'])
    return nb_df

def load_sleap_data(rse, cache_folder, track_root, sessions, RESAMP_INTV):
    slp_bdf_file = oj(cache_folder, 'RRM26-36_sleap_bdf.pq')
    track_df_file = oj(cache_folder, 'RRM26-36_tracks_df.pq')
    rawtrack_file = oj(cache_folder, 'RRM26-36_raw_tracks.pq')
    if os.path.exists(slp_bdf_file) and os.path.exists(track_df_file):
        sleap_bdf = pd.read_parquet(slp_bdf_file).reset_index(drop=True)
        all_track_df = pd.read_parquet(track_df_file).reset_index(drop=True)
        raw_track_df = pd.read_parquet(rawtrack_file).reset_index(drop=True)
    else:
        sleap_bdfs = []
        all_tracks = []
        raw_tracks = []
        for animal in sessions:
            print(animal)
            for day in sessions[animal]:
                session = f'Day{day}'
                print(session)
                try:
                    pose_series = PoseSeries(track_root, animal, session)
                    bmat, neuro_series = rse.load_animal_session(animal, session)
                    bdf = bmat.todf()
                    sleap_bdfs.append(pose_series.sleap_tdf.merge(bdf, 
                                                                on=['animal', 'session', 
                                                                    'trial', 'restaurant'], how='left'))
                    resamp_track_df = pose_series.preprocess_track_df(resample_interval=RESAMP_INTV)
                    all_tracks.append(resamp_track_df)
                    raw_tracks.append(pose_series.track_df)
                except:
                    print(f'Error with {animal} {session}')
        sleap_bdf = pd.concat(sleap_bdfs, axis=0)
        all_track_df = pd.concat(all_tracks, axis=0)  
        raw_track_df = pd.concat(raw_tracks, axis=0)
        sleap_bdf.to_parquet(slp_bdf_file, index=False)
        all_track_df.to_parquet(track_df_file, index=False)
        raw_track_df.to_parquet(rawtrack_file, index=False)
                
    sleap_bdf['bonsai_decision'] = sleap_bdf['accept'].map({1: 'accept', 0: 'reject'})
    sleap_bdf.loc[~sleap_bdf['quit'].isnull(), 'bonsai_decision'] = 'quit'
    return sleap_bdf, all_track_df, raw_track_df


def add_past_outcome_features(data, dt=120, inplace=False): 
    """ Method to add past outcome information as features to data
        data: pd.DataFrame
        dt: float, time periods to include in seconds, by default 120
    """
    if not inplace:
        data = data.reset_index(drop=True)
    data['wait_time'] = np.NaN
    waited_sel = (data['accept'] == 1) & (~data['outcome'].isnull())
    data.loc[waited_sel, 'wait_time'] =  data.loc[waited_sel, 'outcome'] - data.loc[waited_sel, 'choice']
    quit_sel = ~data['quit'].isnull()
    data.loc[quit_sel, 'wait_time'] = data.loc[quit_sel, 'quit'] - data.loc[quit_sel, 'choice']
    col_arg1 = f'past_rew_{dt}s'
    data[col_arg1] = 0.0
    col_arg2 = f'past_accept_{dt}s'
    data[col_arg2] = 0.0
    col_arg3 = f'past_waits_{dt}s'
    data[col_arg3] = 0.0
    for i in range(len(data)):
        t_i = data.loc[i, 'tone_onset']
        s_i = t_i - dt
        n_rew = np.sum((data['outcome'] >= s_i) & (
            data['outcome'] < t_i) & (data['reward'] == 1))
        choice_sel = (data['choice'] >= s_i) & (data['choice'] < t_i)
        n_accept = np.sum(choice_sel & (data['accept'] == 1))
        total_wait = np.sum(data.loc[choice_sel & (data['accept'] == 1), 'wait_time'])
        data.loc[i, col_arg1] = n_rew
        data.loc[i, col_arg2] = n_accept
        data.loc[i, col_arg3] = total_wait
    return data

def get_flip_sessions(data_root, nb_df, nonflp=False):
    flp_sessions = []
    for animal in os.listdir(data_root):
        animal_path = oj(data_root, animal)
        if os.path.isdir(animal_path):
            for session in os.listdir(animal_path):
                session_path = oj(animal_path, session)
                if os.path.isdir(session_path):
                    for f in os.listdir(session_path):
                        if f.endswith('.flp'):
                            flp_sessions.append((animal, session))
                            break
    if nonflp:
        return np.logical_and.reduce([(nb_df['animal'] != animal) | (nb_df['session'] != session) for animal, session in flp_sessions])
    if not nonflp:
        return np.logical_or.reduce([(nb_df['animal'] == animal) & (nb_df['session'] == session) for animal, session in flp_sessions])

def calculate_neur_diffs_RR(nb_df, events, sessions, expr):
    """
    Customary function to calculate differences between the left and right ROI neural responses
    around `events` for all `sessions`
    nb_df: pd.DataFrame
        neurobehavior dataframe
    events: list
        list of behavior events
    sessions: dict
        dictionary describing all animal, sessions, 3 means bilateral recordings
    expr: neurobehavior experiment object
        experiment object used to organize data frames
    """
    expr.nbm.nb_cols, expr.nbm.nb_lag_cols = expr.nbm.parse_nb_cols(nb_df)
    nb_diff_dfs = []

    for animal in sessions:
        for sess in sessions[animal]:
            if sessions[animal][sess] == 3:
                session = f"Day{sess}"
                nb_df_session = nb_df[
                    (nb_df["animal"] == animal) & (nb_df["session"] == session)
                ].reset_index(drop=True)
                nb_df_session["roi"] = nb_df_session["roi"].str.replace("_470nm", "")
                basic_cols = [c for c in nb_df_session.columns if "_neur|" not in c]
                nb_result = nb_df_session.loc[
                    nb_df_session["roi"] == "left", basic_cols
                ].reset_index(drop=True)
                nb_result["roi"] = "diff"
                for event in events:
                    ev_cols = expr.nbm.nb_cols[f"{event}_neur"]
                    leftVs = nb_df_session.loc[
                        nb_df_session["roi"] == "left", ev_cols
                    ].values
                    rightVs = nb_df_session.loc[
                        nb_df_session["roi"] == "right", ev_cols
                    ].values
                    nb_result[ev_cols] = 0
                    nb_result[ev_cols] = leftVs - rightVs
                nb_diff_dfs.append(nb_result)
    nb_diff_df = pd.concat(nb_diff_dfs, axis=0)
    return nb_diff_df

def validate_FP_csv(pse, animal, session, debug=True):
    """ This function validates fiber photometry data csv file, and spits out a new one if needed,
    especially designed for the problem with a werid :, ; in data columns
    """
    data_file = pse.encode_to_filename(animal, session)['FP']
    if data_file is None:
        print(f'{animal} {session} not found')
        return 
    data = pd.read_csv(
        data_file
    )
    rewrite = False
    def check_for_symbols(x):
        return ';' in x or ':' in x
    for c in data.columns:
        if data[c].dtype == object:
            for i in range(len(data[c])):
                if isinstance(data[c].iat[i], str) and check_for_symbols(data[c].iat[i]):
                    rewrite = True
                    data.iloc[i, c] = float(data.loc[i, c].replace(';', '').replace(':', ''))   
    if rewrite:
        if debug:
            print(f'need to rewrite {animal} {session}')
        else:
            data.to_csv(data_file, index=False)
