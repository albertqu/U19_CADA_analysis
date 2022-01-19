from behaviors import *
from peristimulus import *
from neurobehavior_base import *

"""
"""
def trial_feature_lag_function_check():
    # The following 3 methods are equivalent, check by see that the three plots are equivalent
    # Method 1: gtruth without using lag function
    data_root = '/content/drive/MyDrive/WilbrechtLab/U19_project/analysis/ProbSwitch/BSDML_processed'
    pse = PS_Expr(data_root)
    animal, session = 'D1-R35_RV', 'p155'  # "RRM033", 'p188' #(GOOD)# #'RRM031', 'p193'
    # bmat, neuro_series = pse.load_animal_session(animal, session)
    nb_df = pse.align_lagged_view('BSD', ['outcome'], laglist=None, animal_ID=animal, session=session)
    nb_df['correct'] = nb_df['action'] == nb_df['state']
    nb_df['correct'] = nb_df['correct'].astype(np.int)
    laglist = {'correct': {'pre': 7, 'post': 7}}
    nb_df_lagged = pse.nbm.lag_wide_df(nb_df, laglist)
    # nb_df_lagged[['trial', 'action', 'state'] + cor_cols]
    plot_df1 = df_melt_lagged_features(nb_df_lagged, 'correct',
                                       ['animal', 'session', 'session_num', 'trial', 'action', 'state',
                                        'trial_in_block'])
    sns.relplot(data=plot_df1[plot_df1['trial_in_block'] == 0], x='correct_lag', y='correct_value', kind='line')

    # Method 2: use lag function in NBM, same
    laglist = {'correct': {'pre': 7, 'post': 7, 'long': True}}
    plot_df2 = pse.nbm.lag_wide_df(nb_df, laglist)
    sns.relplot(data=plot_df2[plot_df2['trial_in_block'] == 0], x='correct_lag', y='correct_value', kind='line')

    # Method 3: use two lags, neur and behavior, problematic since it averages across repetitive values, use .drop_duplicates to solve
    laglist = {'correct': {'pre': 7, 'post': 7, 'long': True},
               'action': {'pre': 3, 'post': 3},
               'outcome_neur': {'pre': 2, 'post': 3, 'long': True}}
    plot_df3 = pse.nbm.lag_wide_df(nb_df, laglist)
    sns.relplot(
        data=plot_df3[plot_df3['trial_in_block'] == 0].drop_duplicates(['animal', 'session', 'trial', 'correct_lag']),
        x='correct_lag', y='correct_value', kind='line')


def test_FP_preprocessing():
    show = True
    folder_load = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_Raw"
    folder_save = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    animal, session = 'A2A-15B-B_RT', 'p153_FP_LH'
    group = 'A2A'
    files = encode_to_filename(folder_load, animal, session)
    files2 = encode_to_filename(folder_save, animal, session, ['processed', 'FP'])
    matfile, green, red, fp = files2['behavior'], files['green'], files['red'], files2['FP']
    fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                   tags=('DA', 'Ca'), show=False)

    if show:
        tags = ('DA', 'Ca')
        fig, axes = plt.subplots(nrows=len(fp_sigs) * 2, ncols=1, sharex=True, figsize=(20, 10))
        for i in range(len(fp_sigs)):
            axes[i * len(fp_sigs)].plot(fp_times[i], fp_sigs[i], label=tags[i])
            axes[i * len(fp_sigs) + 1].plot(iso_times[i], iso_sigs[i], label='415_' + tags[i])
            axes[i * len(fp_sigs) + 1].legend()
            axes[i * len(fp_sigs)].legend()

    cut_thres = 120500
    # cut_thres = None
    if cut_thres is not None:
        for i in range(len(fp_sigs)):
            isel_fp = fp_times[i] >= cut_thres
            isel_iso = iso_times[i] >= cut_thres
            fp_times[i], fp_sigs[i] = fp_times[i][isel_fp], fp_sigs[i][isel_fp]
            iso_times[i], iso_sigs[i] = iso_times[i][isel_iso], iso_sigs[i][isel_iso]

    pT = 120 * 1000
    N = len(iso_times)
    sig_T = [min(np.min(iso_times[i]), np.min(fp_times[i])) + pT for i in range(N)]
    iso_times_pT = [iso_times[i][iso_times[i] < sig_T[i]] for i in range(N)]
    iso_sigs_pT = [iso_sigs[i][iso_times[i] < sig_T[i]] for i in range(N)]
    fp_times_pT = [fp_times[i][fp_times[i] < sig_T[i]] for i in range(N)]
    fp_sigs_pT = [fp_sigs[i][fp_times[i] < sig_T[i]] for i in range(N)]
    base_DA_pT = signal_filter_visualize(iso_times_pT[0], iso_sigs_pT[0],
                                         fp_times_pT[0], fp_sigs_pT[0], isosbestic=False, buffer=True)
    base_Ca_pT = signal_filter_visualize(iso_times_pT[1], iso_sigs_pT[1],
                                         fp_times_pT[1], fp_sigs_pT[1], isosbestic=False, buffer=True)

    base_DA = signal_filter_visualize(iso_times[0], iso_sigs[0], fp_times[0], fp_sigs[0], isosbestic=False,
                                      buffer=True)
    base_Ca = signal_filter_visualize(iso_times[1], iso_sigs[1], fp_times[1], fp_sigs[1], isosbestic=False,
                                      buffer=True)
    bases = [base_DA, base_Ca]
    f0_method = 'robust'
    dff = [None] * 2

    for i in range(len(fp_sigs)):
        base_i = bases[i][f0_method]
        dff[i] = (fp_sigs[i] - base_i) / base_i

    dff_zscore = [(dff[i] - np.mean(dff[i])) / np.std(dff[i], ddof=1) for i in range(N)]

    row = 'FP'
    col = "A{t-1,t}"
    hue = 'S[4]'
    # rows = ('Rewarded', 'Unrewarded')
    rows = ('DA', 'Ca')
    cols = ('ipsi_switch', 'contra_switch')
    fmt = '{} Pre'
    hues = [fmt.format(i) for i in range(1, 4 + 1)]

    zscore = False
    meas = ('zscore_' if zscore else '') + 'dF/F'
    behaviors = ['outcome']
    time_window = np.arange(-2000, 2001, 50)
    effect_arg = "_".join([e for e in [row, col, hue] if e])
    behavior_arg = "_".join(behaviors)
    neur_type = group if group == 'D1' else 'D2'


    mat = h5py.File(matfile, 'r')
    # Get aligned signals to behaviors
    # default to align first
    behavior_times = np.vstack([get_behavior_times(mat, beh) for beh in behaviors])
    nonan_sel = ~np.any(np.isnan(behavior_times), axis=0)
    behavior_times_nonan = behavior_times[:, nonan_sel]


    # get trial features
    def opt2selgroups(opt):
        # add in different data
        if opt is None:
            return {'all': None}
        if opt == 'FP':
            return {'DA': 0, 'Ca': 1}
        return get_trial_features(mat, opt)

    rsel_groups, csel_groups, hsel_groups = opt2selgroups(row), opt2selgroups(col), \
                                            opt2selgroups(hue)
    mat.close()

    # TODO: ADD caps for multiple behavior time latencies

    rec_sigs = dff_zscore if zscore else dff
    # denoise_times
    denoise_times, denoise_sigs = [None] * N, [None] * N
    for i in range(N):
        denoise_sigs[i], denoise_times[i] = denoise_quasi_uniform(rec_sigs[i], fp_times[i])
    denoise_times_pT = [denoise_times[i][denoise_times[i] < sig_T[i]] for i in range(N)]
    denoise_sigs_pT = [denoise_sigs[i][denoise_times[i] < sig_T[i]] for i in range(N)]

    aligned = [align_activities_with_event(rec_sigs[i], fp_times[i], behavior_times_nonan,
                                           time_window, False) for i in range(N)]
    aligned_denoise = [align_activities_with_event(denoise_sigs[i], denoise_times[i], behavior_times_nonan,
                                           time_window, False) for i in range(N)]

    print(f"valid trial number: {behavior_times_nonan.shape[1]}")
    kth = 0

    if show:
        tags = ('DA', 'Ca')
        fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, figsize=(20, 10))
        for i in range(len(fp_sigs)):

            k_event_time = behavior_times_nonan[0, kth]
            time_sel = (fp_times[i] >= k_event_time + time_window[0]) \
                       & (fp_times[i] <= k_event_time + time_window[-1])
            denoise_time_sel = (denoise_times[i] >= k_event_time + time_window[0]) \
                       & (denoise_times[i] <= k_event_time + time_window[-1])
            axes[i].plot(time_window, aligned[i][kth], label=tags[i] + '_aligned')
            axes[i].plot(time_window, aligned_denoise[i][kth], label=tags[i] + '_aligned_denoised')
            axes[i].plot(fp_times[i][time_sel] - k_event_time, rec_sigs[i][time_sel], label=tags[i] + '_raw')
            axes[i].plot(denoise_times[i][denoise_time_sel] - k_event_time, denoise_sigs[i][denoise_time_sel],
                         label=tags[i] + '_denoised')
            axes[i].legend()
            axes[i].axvline(0)


#######################################################
################# Data Structure Test #################
#######################################################
def test_BehaviorMat_initialize():
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data_new"
    output = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/tests/BehaviorMat"
    if not os.path.exists(output):
        os.makedirs(output)
    animal, session = "A2A-15B-B_RT", "p153_FP_LH"
    processed = encode_to_filename(folder, animal, session, ['processed'])
    hfile = h5py.File(processed)
    bmat = BehaviorMat(animal, session, hfile)
    trial_event_mat = pd.DataFrame(hfile['out/value/trial_event_mat'], columns=['eventcode', 'time', 'trial'])
    trial_event_mat.to_csv(os.path.join(output, f"trial_event_mat_{animal}_{session}.csv"))
    bmatlist = bmat.event_list.tolist()
    bmatcsv = pd.DataFrame([[bn.event, bn.ecode, bn.etime, bn.trial, bn.saliency, bn.MLAT, bn.merged] for
                            bn in bmatlist], columns=['event', 'ecode', 'time', 'trial', 'saliency',
                                                      'MLAT', 'merge'])
    bmatcsv.to_csv(os.path.join(output, f"bmat_{animal}_{session}.csv"))
    complexities = np.vstack([np.arange(1, bmat.trialN+1), bmat.struct_complexity, bmat.exp_complexity]).T

    complex_csv = pd.DataFrame(complexities, columns=['trial', 'structural', 'exploratory'])
    complex_csv.to_csv(os.path.join(output, f"complexity_mat.csv"))

    get_behavior_times(bmat, 'outcome{t-1}', as_df=True).to_csv(os.path.join(output, 'outcome-1.csv'))
    get_behavior_times(bmat, 'side_out__first', as_df=True).to_csv(os.path.join(output, 'sideout_first.csv'))


# Build DataFrame containing all behavior data
import jplotprefs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import re
import matplotlib as mpl
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# rc('text', usetex=False)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (10, 10)  # (w, h)


def get_data_directories(mouse_id, day):
    fp_folder = '/Users/travis/Google Drive/Wilbrecht Lab/Restaurant Row/data/Cohort 2 A2A/fp_data'
    behavior_folder = '/Users/travis/Google Drive/Wilbrecht Lab/Restaurant Row/data/Cohort 2 A2A/rr_data'

    for file in os.listdir(fp_folder):
        day_check = []
        mouse_id_check = []
        if file.endswith('.csv') & file.startswith('FPTS_'):
            # Pull animal ID from filename
            pattern = 'ID-........'
            match = re.findall(pattern, file)
            if not not match:
                mouse_id_check = match[0][3:]
            # Pull RR Day from filename
            pattern = 'Dayp...'
            match = re.findall(pattern, file)
            if not not match:
                day_check = int(match[0][-3:])
            if ((not not mouse_id_check) & (not not day_check)):
                if (mouse_id == mouse_id_check) & (day == int(day_check)):
                    fpts_file = fp_folder + '/' + file
    for file in os.listdir(fp_folder):
        day_check = []
        mouse_id_check = []
        if file.endswith('.csv') & file.startswith('FP_'):
            # Pull animal ID from filename
            pattern = 'ID-........'
            match = re.findall(pattern, file)
            if not not match:
                mouse_id_check = match[0][3:]
            # Pull RR Day from filename
            pattern = 'Dayp...'
            match = re.findall(pattern, file)
            if not not match:
                day_check = int(match[0][-3:])
            if ((not not mouse_id_check) & (not not day_check)):
                if (mouse_id == mouse_id_check) & (day == int(day_check)):
                    fp_file = fp_folder + '/' + file
    for file in os.listdir(behavior_folder):
        day_check = []
        mouse_id_check = []
        if file.endswith('.csv') & file.startswith('RR_'):
            # Pull animal ID from filename
            pattern = 'ID-........'
            match = re.findall(pattern, file)
            if not not match:
                mouse_id_check = match[0][3:]
            # Pull RR Day from filename
            pattern = 'Dayp...'
            match = re.findall(pattern, file)
            if not not match:
                day_check = int(match[0][-3:])
            if ((not not mouse_id_check) & (not not day_check)):
                if (mouse_id == mouse_id_check) & (day == int(day_check)):
                    rr_file = behavior_folder + '/' + file
    print(fp_file)
    return rr_file, fp_file, fpts_file


def get_fp_plots(animal, day, alignment, sg, side, condition):
    # animal = 'A2A18DRV'
    # day = 282
    (rr_file, fp_file, fp_time_stamps) = get_data_directories(animal, day)

    # fp_file = '/Users/travis/Google Drive/Wilbrecht Lab/Restaurant Row/data/Cohort 2 A2A/fp_data/FP_Dayp278_epoch-7_ID-A2A18DRV_2021-01-07T10_40_53.csv'
    # fp_time_stamps = '/Users/travis/Google Drive/Wilbrecht Lab/Restaurant Row/data/Cohort 2 A2A/fp_data/FPTS_Dayp278_epoch-7_ID-A2A18DRV_2021-01-07T10_40_08.csv'
    # rr_file = '/Users/travis/Google Drive/Wilbrecht Lab/Restaurant Row/data/Cohort 2 A2A/rr_data/RR_Dayp278_epoch-7_ID-A2A18DRV2021-01-07T10_40_08.csv'

    data = pd.read_csv(fp_file, skiprows=1, names=[
        'frame', 'cam_time_stamp', 'flag', 'right_red', 'left_red', 'right_green', 'left_green'])
    data_time_stamps = pd.read_csv(
        fp_time_stamps, skiprows=1, names=['time_stamps']) # TODO: del skip_rows=1

    data_fp = pd.concat([data, data_time_stamps.time_stamps], axis=1)
    rr_data = pd.read_csv(rr_file, sep=' ', header=None,
                          names=['time', 'b_code', 'none'])

    # Classify events and add class to data_rr df
    (reject_events,
     accept_and_rewarded_events,
     num_accept_rewarded_events,
     quit_events, num_quit_events,
     pct_no_offer_rejects,
     data_rr) = classify_events(rr_data)

    # Green signal
    right_green_fp = data_fp.right_green[data_fp.flag == 2].values
    right_green_fp_ts = data_fp.time_stamps[data_fp.flag == 2].values
    left_green_fp = data_fp.left_green[data_fp.flag == 2].values
    left_green_fp_ts = data_fp.time_stamps[data_fp.flag == 2].values
    # Red signal
    right_red_fp = data_fp.right_red[data_fp.flag == 4].values
    right_red_fp_ts = data_fp.time_stamps[data_fp.flag == 4].values
    left_red_fp = data_fp.left_red[data_fp.flag == 4].values
    left_red_fp_ts = data_fp.time_stamps[data_fp.flag == 4].values

    # Control signal (415nm)
    right_control_fp = data_fp.right_green[data_fp.flag == 1].values
    right_control_fp_ts = data_fp.time_stamps[data_fp.flag == 1].values
    left_control_fp = data_fp.left_green[data_fp.flag == 1].values
    left_control_fp_ts = data_fp.time_stamps[data_fp.flag == 1].values

    # Calculate time window for plotting FP data
    WINDOW_S = 2  # number of seconds before and after event to plot FP data
    frame_interval = np.nanmean(np.diff(right_red_fp_ts)) / 1000
    time_window = int(WINDOW_S / frame_interval)

    # condition can be "reject","rewarded" or "quit"
    plot_trace_probs(alignment, sg, side, condition, data_fp, data_rr, time_window, frame_interval)


def plot_trace_probs(alignment, sg, side, condition, data_fp, data_rr, time_window, frame_interval):
    events = {
        'reward': [16, 28, 40, 52],
        # Servo arm open (should track with pellet taken fro dispenser)
        'servo_open': [1, 3, 5, 7],
        'reward_omission': [15, 27, 39, 51],
        'offer_tone_0': [17, 29, 41, 53],  # no-reward tone codes
        'offer_tone_20': [18, 30, 42, 54],
        'offer_tone_80': [19, 31, 43, 55],  # 80pct rewarded tone codes
        'offer_tone_100': [20, 32, 44, 56],  # reward tone codes
        'any_offer': [18, 19, 20, 30, 31, 32, 42, 43, 44, 54, 55, 56],
        'exit': [63, 66, 69, 72],
        'entry': [61, 64, 67, 70],
        'accept': [62, 65, 68, 71]
    }
    if alignment == 'reject':
        [num_rejects, num_no_reward_tones,
         reject_ts] = count_rejections(data_rr)
        event_ts = reject_ts
    else:
        event_codes = events.get(alignment)
    if side == 'left':
        if sg == 'green':
            signal_fp = data_fp.left_green[data_fp.flag == 2].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 2].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
        elif sg == 'red':
            signal_fp = data_fp.left_red[data_fp.flag == 4].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 4].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
        elif sg == 'control':
            signal_fp = data_fp.left_green[data_fp.flag == 1].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 1].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
    if side == 'right':
        if sg == 'green':
            signal_fp = data_fp.right_green[data_fp.flag == 2].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 2].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
        elif sg == 'red':
            signal_fp = data_fp.right_red[data_fp.flag == 4].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 4].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
        elif sg == 'control':
            signal_fp = data_fp.right_green[data_fp.flag == 1].values
            signal_fp_ts = data_fp.time_stamps[data_fp.flag == 1].values
            signal_fp_ts = signal_fp_ts[~np.isnan(signal_fp_ts)]  # remove nan
    ax_index = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for rr in [1, 2, 3, 4]:
        if alignment != 'reject':
            event_code = event_codes[rr - 1]
            event_idx = data_rr.b_code[data_rr.b_code ==
                                       event_code].index.tolist()
            condition_matched = np.array([])
            # Filter for events that match condition: 'reject', 'rewarded', 'quit'
            # if np.sum(event_idx) > 0:
            #    for event in event_idx:
            #        if (data_rr.event_class[event] == condition):
            #            condition_matched = np.append(condition_matched, event)
            # event_idx = condition_matched
            event_ts = data_rr.time[event_idx].values

        for prob in [0, 20, 80, 100]:
            traces = np.zeros([len(event_ts), time_window * 2])
            for i in np.arange(0, len(event_ts), 1):
                if data_rr.offer_tone[event_idx[i]] == prob:
                    ts_rr = event_ts[i]
                    ts_fp = np.argmax(signal_fp_ts > ts_rr)  # DOUBLE CHECK THIS
                    if (ts_fp > time_window) & ((ts_fp + time_window) < len(signal_fp)):
                        trace = signal_fp[ts_fp - time_window:ts_fp + time_window]
                        traces[i, :] = trace - trace[0]
            t = np.arange(-time_window, time_window, 1) * frame_interval
            mean_trace = np.mean(traces, axis=0)
            sem_trace = np.std(traces, axis=0) / np.sqrt(len(traces))
            extent = [min(t), max(t), 0, 1]
            # axes[ax_index[rr-1]].imshow(traces,extent=extent)
            axes[ax_index[rr - 1]].plot(t, mean_trace,
                                        label=str(prob) + '% tone')
            axes[ax_index[rr - 1]].fill_between(t, mean_trace + sem_trace,
                                                mean_trace - sem_trace, alpha=0.5)
            ymin = -0.5e-5
            ymax = 5e-5
            axes[ax_index[rr - 1]].plot([0, 0], [ymin, ymax], '--k')
            axes[ax_index[rr - 1]].set_xlabel('Time (s)')
            axes[ax_index[rr - 1]].set_ylabel('FL Signal (a.u)')
            # axes[ax_index[rr-1]].title(alignment + ' ' + sg + ', R'+str(rr))
            axes[ax_index[rr - 1]].set_title('R' + str(rr))
            axes[ax_index[rr - 1]].legend()
            # axes[ax_index[rr-1]].set_ylim([-.00004,.00005])
            axes[ax_index[rr - 1]].set_ylim(ymin, ymax)
    fig_title = alignment + ' ' + side + ' ' + sg + ' ' + condition
    plt.suptitle(fig_title)
    plt.tight_layout()


def classify_events(df):
    # This will find timestamps and count where all the "clean" rejections occur.
    # By this, we mean the mouse hears offer tone and completely skips the restaurant without entering it.
    data_rr = df.assign(event_class=np.ones(len(df)) * np.nan)  # Add 'event_class' column
    data_rr = df.assign(offer_tone=np.ones(
        len(df)) * np.nan)  # Add 'offer_tone' column to indicate which tone was given for each event
    reward_codes_0 = [17, 29, 41, 53]  # no-reward tone codes
    reward_codes_20 = [18, 30, 42, 54]  # 20pct rewarded tone codesc
    reward_codes_80 = [19, 31, 43, 55]  # 80pct rewarded tone codes
    reward_codes_100 = [20, 32, 44, 56]  # reward tone codes
    reward_taken_codes = [16, 28, 40, 52]  # Pellet taken from dispenser
    # Servo arm open (should track with pellet taken fro dispenser)
    servo_open_codes = [1, 3, 5, 7]
    exit_codes = [63, 66, 69, 72]  # Exit codes, aka "Sharp" timestamps
    entry_codes = [61, 64, 67, 70]  # Entry codes, aka "Sharp"
    accept_codes = [62, 65, 68, 71]  # Sharp accept codes
    # Data frame initialization for holding sorted event timestamps
    reject_events = pd.DataFrame(columns=['reject_tone_ts', 'reject_exit_ts', 'restaurant'])
    num_no_offer_rejects = 0
    accept_and_rewarded_events = pd.DataFrame(columns=['tone_ts', 'accept_ts', 'restaurant'])
    num_accept_rewarded_events = 0
    accept_not_rewarded_events = pd.DataFrame(columns=['tone_ts', 'accept_ts', 'restaurant'])
    num_accept_not_rewarded_events = 0
    quit_events = pd.DataFrame(columns=['tone_ts', 'quit_ts', 'restaurant'])
    num_quit_events = 0

    for rr in [1, 2, 3, 4]:
        offer_tone_100_idx = df.index[df.b_code.isin([reward_codes_100[rr - 1]])].values
        print(len(offer_tone_100_idx))
        offer_tone_80_idx = df.index[df.b_code.isin([reward_codes_80[rr - 1]])].values
        offer_tone_20_idx = df.index[df.b_code.isin([reward_codes_20[rr - 1]])].values
        offer_tone_0_idx = df.index[df.b_code.isin([reward_codes_0[rr - 1]])].values
        num_no_offers = len(offer_tone_0_idx)
        tone_idx = np.append(offer_tone_100_idx, offer_tone_80_idx)
        tone_idx = np.append(tone_idx, offer_tone_20_idx)
        tone_idx = np.append(tone_idx, offer_tone_0_idx)
        accept_idx = df.index[df.b_code.isin([accept_codes[rr - 1]])].values
        exit_idx = df.index[df.b_code.isin([exit_codes[rr - 1]])].values
        entry_idx = df.index[df.b_code.isin([entry_codes[rr - 1]])].values
        reward_taken_idx = df.index[
            df.b_code.isin([reward_taken_codes[rr - 1]])].values  # Pellet taken from dispenser
        # Servo arm open (should track with pellet taken fro dispenser)
        servo_open_idx = df.index[df.b_code.isin([servo_open_codes[rr - 1]])].values
        print('Pellets Revealed R' + str(rr) + ': ' + str(len(servo_open_idx)))
        print('Pellets Eaten R' + str(rr) + ': ' + str(len(reward_taken_idx)))
        for event in tone_idx:
            # Determine which offer tone was given for each event
            code = df.b_code[event]
            if code in reward_codes_0:
                tone_prob = 0
            if code in reward_codes_20:
                tone_prob = 20
            if code in reward_codes_80:
                tone_prob = 80
            if code in reward_codes_100:
                tone_prob = 100
            # make sure events occurs after tone (e.g. not last unfinished trial)
            if (np.any(entry_idx > event) & np.any(exit_idx > event) & np.any(servo_open_idx > event)):
                next_entry_idx = min(entry_idx[entry_idx > event])
                next_accept_idx = min(accept_idx[accept_idx > event])
                next_exit_idx = min(exit_idx[exit_idx > event])
                next_pellet_reveal_idx = min(servo_open_idx[servo_open_idx > event])
                # print('Tone: '+ str(event))
                # print('Entry: '+str(next_entry_idx))
                # print('Accept: '+str(next_accept_idx))
                # print('Pellet taken: '+str(next_pellet_reveal_idx))
                # print('Exit: '+str(next_exit_idx))
                # Reject Events
                if next_exit_idx < next_accept_idx:
                    # print('Reject')
                    reject_tone_ts = df.time[event]
                    reject_exit_ts = df.time[next_exit_idx]
                    reject_events = reject_events.append(
                        {'reject_tone_ts': reject_tone_ts, 'reject_exit_ts': reject_exit_ts,
                         'restaurant': rr}, ignore_index=True)
                    if event in offer_tone_0_idx:
                        num_no_offer_rejects += 1
                    data_rr.loc[event, 'event_class'] = 'reject'
                    data_rr.loc[next_entry_idx, 'event_class'] = 'reject'
                    data_rr.loc[next_accept_idx, 'event_class'] = np.nan
                    data_rr.loc[next_exit_idx, 'event_class'] = 'reject'
                    data_rr.loc[next_pellet_reveal_idx, 'event_class'] = 'reject'
                    data_rr.loc[event, 'offer_tone'] = tone_prob
                    data_rr.loc[next_entry_idx, 'offer_tone'] = tone_prob
                    data_rr.loc[next_accept_idx, 'offer_tone'] = tone_prob
                    data_rr.loc[next_exit_idx, 'offer_tone'] = tone_prob
                    data_rr.loc[next_pellet_reveal_idx, 'offer_tone'] = tone_prob

                # Accept_rewarded events
                if (next_pellet_reveal_idx < next_exit_idx):
                    # print('Accept')
                    accept_tone_ts = df.time[event]
                    accept_event_ts = df.time[next_accept_idx]
                    accept_and_rewarded_events = accept_and_rewarded_events.append(
                        {'tone_ts': accept_tone_ts, 'accept_ts': accept_event_ts, 'restaurant': rr},
                        ignore_index=True)


