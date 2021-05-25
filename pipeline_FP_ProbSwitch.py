from peristimulus import *
from behaviors import *
from utils import encode_to_filename, decode_from_filename
try:
    from caiman.utils.utils import recursively_save_dict_contents_to_group
except ModuleNotFoundError:
    print("CaImAn not installed or environment not activated, certain functions might not be usable")


#######################################################
######################### FP ##########################
#######################################################
# TODO: unify std ddof
def FP_save():
    # hue: ITI, row: trial history, col: laterality
    show = False
    folder_load = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_Raw"
    folder_save = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    #choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    choices = get_prob_switch_all_sessions(folder_save, {'A2A': '*', 'D1': '*'})
    # TODO: ADD base method
    zscore = False # Should not matter with 1 session
    base_methods = ['robust_fast', 'mode', 'perc15'] # has to be list
    tags = ('DA', 'Ca')
    overwrite = True
    test_only = False

    sample = 120 * 1000

    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        sessions = choices[group]
        for animal in sessions:
            for session in sessions[animal]:
                print(animal, session)
                files = encode_to_filename(folder_load, animal, session)
                files2 = encode_to_filename(folder_save, animal, session)

                green, red = files['green'], files['red']
                fp, matfile = files2['FP'], files2['processed']
                # Load FP
                if fp is None or not check_FP_contain_dff_method(fp, base_methods) or overwrite:
                    aux_times = None
                    mat = h5py.File(matfile)

                    if 'out' in mat:  # Processed data
                        assert mat['out/value/green_time'].shape[0] == 1
                        aux_times = [np.array(mat['out/value/green_time']).ravel(),
                                     np.array(mat['out/value/red_time']).ravel()]
                    fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                                   aux_times=aux_times,
                                                                                tags=('DA', 'Ca'), show=False)

                    if not test_only:
                        fp_dffs = {b: [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i],
                                                        b, zscore=False) for i in range(len(fp_sigs))]
                                   for b in base_methods if not check_FP_contain_dff_method(fp, b)}
                        p = os.path.join(folder_save, f'{animal}_{session}')

                        try:
                            fp_hdf5 = h5py.File(os.path.join(p, f"{animal}_{session}.hdf5"),
                                                'w' if overwrite else ('w-' if fp is None else 'a'))
                            to_save = {tags[i]: {'dff': {b: fp_dffs[b][i] for b in fp_dffs},
                                                 'time': fp_times[i]} for i in range(len(tags))}
                            recursively_save_dict_contents_to_group(fp_hdf5, '/', to_save)
                            fp_hdf5.close()

                        except IOError:
                            print("OOPS!: The file already existed ease try with another file, "
                                  "new results will NOT be saved")
                            continue

                    if show:
                        fig, axes = plt.subplots(nrows=len(fp_sigs) * 2, ncols=1, sharex=True,figsize=(20,10))
                        for i in range(len(fp_sigs)):
                            axes[i * len(fp_sigs)].plot(fp_times[i], fp_sigs[i], label=tags[i])
                            axes[i * len(fp_sigs) + 1].plot(iso_times[i], iso_sigs[i], label='415_' + tags[i])
                            axes[i * len(fp_sigs) + 1].legend()
                            axes[i * len(fp_sigs)].legend()
                        subfolder = os.path.join(plots, 'FP_test')
                        if not os.path.exists(subfolder):
                            os.makedirs(subfolder)
                        fig.savefig(os.path.join(subfolder, f'{animal}_{session}_FP_channel_split.png'))
                        plt.close(fig)

                        if sample is not None:
                            fig, axes = plt.subplots(nrows=len(fp_sigs) * 2, ncols=1, sharex=True,
                                                     figsize=(20, 10))
                            for i in range(len(fp_sigs)):
                                sel = (fp_times[i] < (np.min(fp_times[i]) + sample))
                                axes[i * len(fp_sigs)].plot(fp_times[i][sel], fp_sigs[i][sel],
                                                            label=tags[i])
                                sel2 = (iso_times[i] < (np.min(iso_times[i]) + sample))
                                axes[i * len(fp_sigs) + 1].plot(iso_times[i][sel2], iso_sigs[i][sel2],
                                                                label='415_' + tags[i])
                                axes[i * len(fp_sigs) + 1].legend()
                                axes[i * len(fp_sigs)].legend()
                            subfolder2 = os.path.join(subfolder, f'sample{sample // 1000}')
                            if not os.path.exists(subfolder2):
                                os.makedirs(subfolder2)
                            fig.savefig(os.path.join(subfolder2, f'{animal}_{session}_FP_channel_split.png'))
                            plt.close(fig)


#######################################################
######################### FP ##########################
#######################################################
def raw_trace_visualization(folder, animal, session, isel=None, zscore=True, base_method='robust_fast'):
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_probswitch_raw"
    time_window_tuple = (1000, 1000, 50)
    files = encode_to_filename(folder, animal, session)
    group = animal.split("_")[0]
    neur = group if group == 'D1' else 'D2'
    sigs = ('DA', 'Ca')
    meas = ('zscore_' if zscore else '') + 'dF/F'

    matfile, green, red, fp = files['processed'], files['green'], files['red'], files['FP']
    mat = h5py.File(matfile, 'r')
    # Load FP
    if fp is not None:
        with h5py.File(fp, 'r') as fp_hdf5:
            fp_sigs = [access_mat_with_path(fp_hdf5, f'{sigs[i]}/dff/{base_method}')
                       for i in range(len(sigs))]
            if zscore:
                fp_sigs = [(fp_sigs[i] - np.mean(fp_sigs[i])) / np.std(fp_sigs[i], ddof=1)
                           for i in range(len(fp_sigs))]
            fp_times = [access_mat_with_path(fp_hdf5, f'{sigs[i]}/time') for i in range(len(sigs))]
    else:
        print(f"Warning {animal} {session} does not have photometry processed!")
        fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                       tags=('DA', 'Ca'), show=False)

        fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                    zscore=zscore) for i in range(len(fp_sigs))]

    behaviors = ('center_in', 'center_out', 'choice', 'outcome', 'side_out')
    outcomes = ('Incorrect', 'Correct Omission', 'Rewarded')
    lateralities = ('ipsi', 'contra')
    behavior_times = {b: get_behavior_times(mat, b)[0] for b in behaviors}
    N_trial = len(behavior_times[behaviors[0]])
    trials = np.arange(N_trial)
    if isel is None:
        isel = trials
    nonan2_sels = np.zeros(N_trial, dtype=bool)
    event_times = np.vstack([behavior_times['center_in'][:-2], behavior_times['outcome'][2:]])
    nonan2_sels[2:] = (~np.isnan(behavior_times['center_in'][:-2]))&(~np.isnan(behavior_times['outcome'][2:]))
    trial_outcomes = get_trial_outcomes(mat)
    trial_laterality = get_trial_outcome_laterality(mat)
    itis = access_mat_with_path(mat, "glml/trials/ITI", ravel=True)
    intervals = [(1.05, 4), (0.65, 1.05), (0.5, 0.65), (0, 0.5)]
    itis_binned = {itvl: (itis > itvl[0]) & (itis <= itvl[1]) for itvl in intervals}
    two_trials = get_trial_features(mat, "O{t-2,t-1}")
    side_stay = get_trial_features(mat, "A{t-1,t}")
    mat.close()
    # two_trials = {}
    # for oi in outcomes:
    #     for oj in outcomes:
    #         oarr = np.zeros(N_trial, dtype=bool)
    #         oarr[2:] = (trial_outcomes[oi][:-2] & trial_outcomes[oj][1:-1])
    #         two_trials[oi[0]+oj[0]] = oarr
    # side_stay = {}
    # for il in lateralities:
    #     for jl in lateralities:
    #         stay = 'stay' if (il == jl) else 'switch'
    #         larr = np.zeros(N_trial, dtype=bool)
    #         larr[2:] = trial_laterality[il][2:] & trial_laterality[jl][1:-1]
    #         side_stay[jl + '_' + stay] = larr

    palettes = sns.color_palette()
    for twot in two_trials:
        # note switch stay
        for sstay in side_stay:
            tts_sel = nonan2_sels & two_trials[twot] & side_stay[sstay]
            tts = trials[tts_sel]
            for tt in tts:
                pasttwo, lats = [None] * 3, [None] * 3
                for ii in range(3):
                    for oii in outcomes:
                        if trial_outcomes[oii][tt-2+ii]:
                            pasttwo[ii] = oii
                    for lati in lateralities:
                        if trial_laterality[lati][tt-2+ii]:
                            lats[ii] = lati
                itibin = None
                for iiti, iti in enumerate(itis_binned):
                    if itis_binned[iti][tt]:
                        itibin = iiti
                putative_iti = behavior_times['center_in'][tt] - behavior_times['side_out'][tt-1]
                flag = False
                if (pasttwo[-3][0] + pasttwo[-2][0] != twot) or ((lats[-2]+
                                            ('_stay' if lats[-1] != lats[-2] else '_switch')) == sstay):
                    print(tt, pasttwo[-3][0] + pasttwo[-2][0], (lats[-2] + ('_stay' if lats[-1] == lats[-2]
                                                                        else '_switch')))
                    flag = True
                elif (np.abs(putative_iti/1000 - itis[tt]) > 0.001):
                    print(tt, np.abs(putative_iti / 1000 - itis[tt]), putative_iti / 1000, itis[tt], itibin)
                    flag = True
                else:
                    pass


                ev_tms = np.array([[behavior_times['center_in'][tt-2], behavior_times['side_out'][tt]]]).T
                if flag:
                    if tt in isel or isel == tt:
                        print("There we go")
                        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 10))
                        for k, fsig in enumerate(sigs):
                            t_window = calculate_best_time_window(fp_times[k], ev_tms, time_window_tuple,
                                                                  align_last=True)

                            aligned_trial = align_activities_with_event(fp_sigs[k], fp_times[k], ev_tms, t_window,
                                                       discrete=False, align_last=True).ravel()
                            axes[k].plot(t_window, aligned_trial, color=palettes[-k], label='signal')
                            axes[k].set_ylabel(fsig + '_' + meas)
                            axes[k].set_xlabel('time(ms)')
                            curr=None
                            for outc in outcomes:
                                if trial_outcomes[outc][tt]:
                                    curr = outc
                                    break

                            for i, beh in enumerate(behavior_times):
                                for j, v in enumerate(behavior_times[beh][tt-2:tt+1]):
                                    align_v = v - ev_tms.ravel()[-1]
                                    if beh == 'outcome' and j == 1:
                                        axes[k].text(align_v, 0, 'O(t-1)')
                                    if beh == 'side_out':
                                        axes[k].text(align_v +100, 0, 'ITI')
                                    if j == 0:
                                        axes[k].axvline(v - ev_tms.ravel()[-1], color=palettes[i+1], label=beh)
                                    else:
                                        axes[k].axvline(v-ev_tms.ravel()[-1], color=palettes[i+1])
                            axes[k].legend()
                        fig.suptitle(f"{neur} trialHist:{twot}, ITI{itibin}, action:{sstay}, trial {tt+1}, "
                                     f"current:{curr}")
                        plt.tight_layout()
                        fname = f"trial{tt+1}_trialHist_{twot}_ITI{itibin}_action_{sstay}_{animal}_{session}"
                        fname = os.path.join(plots, "iti_debug", fname)
                        #fig.savefig(fname + '.png')
                        fig.savefig(fname + '.eps')
                        plt.close(fig)



#######################################################
################# NAc D1/D2 FP Outcome ################
#######################################################

def NAcD1D2_trialHist_ITI(choices=None):
    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    #choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    if choices is None:
        choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH", "p238_FP_LH"],
                           'A2A-19B_LT': ['p159_FP_LH'],
                           'A2A-19B_RT': ['p139_FP_LH', 'p148_FP_LH'],
                           'A2A-19B_RV': ['p142_FP_RH', 'p147_FP_LH', 'p156_FP_LH']
                           }, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    zscore = True # Should not matter with 1 session
    base_method = 'robust'
    denoise=True


    # Plotting Option
    sigs = ('DA', 'Ca')
    row = "R{t-2,t-1}"
    col = "A{t-1,t}"
    hue = "ITI"
    rows = ("UU", "UR", "RR", "RU")
    cols = ("ipsi_stay", "ipsi_switch", "contra_stay", "contra_switch")
    hues = [(1.05, 4), (0.65, 1.05), (0.5, 0.65), (0, 0.5)]
    #hues = [(0.65, 1.05), (0.5, 0.65)]
    plot_type = 'trial_average'

    options = {'sigs': sigs, 'row': row, 'rows': rows,
               'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
    behavior_aligned_FP_plots(folder, plots, 'outcome{t-1}', choices, options, zscore, base_method, denoise)
    behavior_aligned_FP_plots(folder, plots, 'outcome{t-2}', choices, options, zscore, base_method, denoise)
    behavior_aligned_FP_plots(folder, plots, 'center_in', choices, options, zscore, base_method, denoise)
    behavior_aligned_FP_plots(folder, plots, 'center_in{t-1}', choices, options, zscore, base_method, denoise)
    behavior_aligned_FP_plots(folder, plots, 'center_out', choices, options, zscore, base_method, denoise)
    behavior_aligned_FP_plots(folder, plots, 'side_out{t-1}', choices, options, zscore, base_method, denoise)


def NAcD1D2_CADA_outcome(choices=None, plot_type='trial_average'):
    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    #choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    # if choices is None:
    #     choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH", "p238_FP_LH"],
    #                        'A2A-19B_RT': ['p139_FP_LH', 'p148_FP_LH'],
    #                        'A2A-19B_RV': ['p142_FP_RH', 'p156_FP_LH']
    #                        },
    #                "D1": {"D1-27H_LT": ["p103_FP_RH", "p189_FP_RH"],
    #                       "D1-28B_LT": ["p135_session2_FP_LH"]}}
    zscore = True # Should not matter with 1 session
    # base_method = 'robust'
    # denoise = True  # denoise
    for sg in ['Ca', 'DA']:
        choices = get_probswitch_session_by_condition(folder, group='all', region='NAc', signal=sg)
        for denoise in [True, False]:
            for base_method in ['robust_fast', 'perc15', 'mode']:
                # Plotting Option
                sigs = [sg]
                row = "FP"
                col = "A"
                hue = "O"
                rows = (sg, )
                cols = ('ipsi', 'contra')
                hues = ('Incorrect', 'Correct Omission', 'Rewarded')
                ylims = [[(-1.5, 2.1)] * 2]

                options = {'sigs': sigs, 'row': row, 'rows': rows, 'ylim': ylims,
                           'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
                behavior_aligned_FP_plots(folder, plots, 'outcome', choices, options,
                                          zscore, base_method, denoise)


def DMSD1D2_CADA_all_phase(choices=None, plot_type='trial_average'):
    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_DMS_D1D2_CADA"
    choices = get_probswitch_session_by_condition(folder, group='all', region='DMS', signal='all')
    zscore = True # Should not matter with 1 session
    base_method = 'robust_fast'
    # PERISWITCH


    for sg in ['Ca', 'DA']:
        choices = get_probswitch_session_by_condition(folder, group='all', region='DMS', signal=sg)
        #choices = {'A2A': {}, 'D1': {"D1-27H_LT": ['p105_FP_LH']}}
        for denoise in [True, False]:
            #for base_method in ['robust_fast', 'perc15', 'mode']:
            for base_method in ['robust_fast']:
                # Plotting Option
                sigs = [sg]
                row = "FP"
                #col = "A"
                col = "A{t-1,t}"
                #hue = "O"
                hue = None
                rows = (sg, )
                #cols = ('ipsi', 'contra')
                cols = ('ipsi_stay', 'contra_stay', 'ipsi_switch', 'contra_switch')
                #hues = ('Incorrect', 'Correct Omission', 'Rewarded')
                hues=None

                ylimMap = {
                    'outcome': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                    'center_out': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                   'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                    'side_out': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                 'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                    'side_out{t-1}': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                 'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                    'choice': {'Ca': None, 'DA': None},
                    'center_in': {'Ca': None, 'DA': None}}

                #for behavior in ['center_in', 'center_out', 'choice', 'outcome', 'side_out']:
                for behavior in ['side_out{t-1}']:
                    ylims = ylimMap[behavior][sg]

                    options = {'sigs': sigs, 'row': row, 'rows': rows, 'ylim': ylims,
                               'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
                    behavior_aligned_FP_plots(folder, plots, behavior, choices, options,
                                              zscore, base_method, denoise)

def NAcD1D2_CADA_all_phase(choices=None, plot_type='trial_average'):
    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    #choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    if choices is None:
        choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH", "p238_FP_LH"],
                           'A2A-19B_RT': ['p139_FP_LH', 'p148_FP_LH'],
                           'A2A-19B_RV': ['p142_FP_RH', 'p156_FP_LH']
                           },
                   "D1": {"D1-27H_LT": ["p103_FP_RH", "p189_FP_RH"],
                          "D1-28B_LT": ["p135_session2_FP_LH"]}}
    zscore = True # Should not matter with 1 session
    base_method = 'robust_fast'
    for denoise in [True, False]:
        # Plotting Option
        sigs = ['all']
        row = "FP"
        col = "A"
        hue = "O"
        rows = ('DA', 'Ca')
        cols = ('ipsi', 'contra')
        hues = ('Incorrect', 'Correct Omission', 'Rewarded')

        options = {'sigs': sigs, 'row': row, 'rows': rows,
                   'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
        for behavior in ['center_in', 'center_out', 'choice', 'outcome', 'side_out']:
            behavior_aligned_FP_plots(folder, plots, behavior, choices, options, zscore, base_method, denoise)


def DMSD1D2_CADA_Periswitch(choices=None, plot_type='trial_average'):
    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_DMS_D1D2_CADA"
    choices = get_probswitch_session_by_condition(folder, group='all', region='DMS', signal='all')
    zscore = True # Should not matter with 1 session
    base_method = 'robust_fast'
    LEN = 4
    # PERISWITCH

    for sg in ['Ca', 'DA']:
        #choices = get_probswitch_session_by_condition(folder, group='all', region='DMS', signal=sg)
        #choices = {'A2A': {'A2A-16B-1_RT': ['p221_FP_RH']}, 'D1': {"D1-27H_LT": ['p102_FP_LH']}}
        choices = {'A2A': {}, 'D1': {"D1-27H_LT": ['p105_FP_LH']}}
        for denoise in [True, False]:
            #for base_method in ['robust_fast', 'perc15', 'mode']:
            for base_method in ['robust_fast']:
                # Plotting Option
                sigs = [sg]
                for hh, hue in enumerate((f"S[-{LEN}]", f"S[{LEN}]")):
                    row = 'R'#'FP'
                    col = "A"
                    rows = ('Rewarded', 'Unrewarded')
                    #rows = (sg,)
                    cols = ('ipsi', 'contra')
                    fmt = '{} Pre' if hh else '{} Post'
                    hues = [fmt.format(i) for i in range(0, LEN+1)]
                    # ylimMap = {'outcome': {'Ca': [[(-1.0, 1.5)] * len(cols)] * len(rows),
                    #                        'DA': [[(-1.6, 1.8)] * len(cols)] * len(rows)},
                    #            'center_out': {'Ca': [[(-1.7, 1.7)] * len(cols)] * len(rows),
                    #                           'DA': [[(-1.5, 1.2)] * len(cols)] * len(rows)}}
                    ylimMap = {
                        'outcome': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                    'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                        'center_out': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                       'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                        'side_out': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                     'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                        'side_out{t-1}': {'Ca': [[(-2, 2)] * len(cols)] * len(rows),
                                          'DA': [[(-2.5, 2.5)] * len(cols)] * len(rows)},
                        'choice': {'Ca': None, 'DA': None},
                        'center_in': {'Ca': None, 'DA': None}}

                    for behavior in ['outcome']:
                        ylims = ylimMap[behavior][sg]
                        #ylims = None

                        options = {'sigs': sigs, 'row': row, 'rows': rows, 'ylim': ylims,
                                   'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
                        behavior_aligned_FP_plots(folder, plots, behavior, choices, options,
                                                  zscore, base_method, denoise)


def NAcD1D2_CADA_outcome_raw():

    # file option
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    #choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"], 'A2A-19B_LT': ['p159_FP_LH'],
                       'A2A-19B_RV': ['p156_FP_LH']},
               "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    zscore = True # Should not matter with 1 session
    denoise = True # denoise
    base_method = 'robust'

    # Plotting Option
    sigs = ['all']
    row = "FP"
    col = "A"
    hue = "O"
    rows = ('DA', 'Ca')
    cols = ('ipsi', 'contra')
    hues = ('Incorrect', 'Correct Omission', 'Rewarded')
    plot_type = 'trial_raw'

    options = {'sigs': sigs, 'row': row, 'rows': rows,
               'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
    behavior_aligned_FP_plots(folder, plots, ['outcome', 'side_out'], choices, options, zscore,
                              base_method, denoise)


def NAcD1D2_Fig3_group():
    # TODO: add functionality for markers
    trial_heatmap_D1D2(center_in_func, "center_in")


def outcome_func(mat, trial_av=False):
    if trial_av:
        outcome_times = get_behavior_times(mat, 'outcome')
        nonan_sel = ~np.isnan(outcome_times)
        return outcome_times[nonan_sel], nonan_sel
    outcome_times = get_behavior_times(mat, 'outcome')
    outcome_trials = np.where(~np.isnan(outcome_times))[0]
    return outcome_times, outcome_trials


def center_in_func(mat, trial_av=False):

    center_in_times = access_mat_with_path(mat, 'glml/time/center_in', ravel=True)
    center_in_trials = access_mat_with_path(mat, 'glml/trials/center_in', ravel=True,
                                            dtype=np.int)-1
    if trial_av:
        if 'glml' in mat:
            mat = access_mat_with_path(mat, "glml", raw=True)
        k = np.prod(access_mat_with_path(mat, 'trials/ITI').shape)
        center_in_times_f = np.full(k, np.nan)
        center_in_times_f[center_in_trials] = center_in_times
    return center_in_times, center_in_trials


def choice_func(mat, trial_av=False):
    if trial_av:
        choice_times = get_behavior_times(mat, 'choice')
        nonan_sel = ~np.isnan(choice_times)
        return choice_times[nonan_sel], nonan_sel
    choice_times = get_behavior_times(mat, 'choice')
    choice_trials = np.where(~np.isnan(choice_times))[0]
    return choice_times, choice_trials


def trial_heatmap_D1D2(event_time_func, tag):
    show = False
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    choices = {'A2A': {'A2A-15B-B_RT': ["p153_FP_LH"]}, "D1": {"D1-27H_LT": ["p103_FP_RH"]}}
    zscore = False # Should not matter with 1 session
    base_method = 'robust'
    two_sessions = []
    # sessions = {'D1': {'D1-27k': ['p107']}, 'A2A':{}}
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row', figsize=(20, 10))
    for j, group in enumerate(['D1', 'A2A']):
        sessions = choices[group]
        time_window = np.arange(-2000, 2001, 50)
        for ia, animal in enumerate(sessions):
            for ks, session in enumerate(sessions[animal]):
                two_sessions.append(animal+"_"+session)
                files = encode_to_filename(folder, animal, session)
                matfile, green, red = files['processed'], files['green'], files['red']
                # Load FP
                fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                            tags=('DA', 'Ca'), show=False)

                fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                           zscore=zscore) for i in range(len(fp_sigs))]
                if show:
                    tags = ('DA', 'Ca')
                    fig, axes = plt.subplots(nrows=len(fp_sigs) * 2, ncols=1, sharex=True)
                    for i in range(len(fp_sigs)):
                        axes[i * len(fp_sigs)].plot(fp_times[i], fp_sigs[i], label=tags[i])
                        axes[i * len(fp_sigs) + 1].plot(iso_times[i], iso_sigs[i], label='415_' + tags[i])
                        axes[i * len(fp_sigs) + 1].legend()
                        axes[i * len(fp_sigs)].legend()
                    # TODO: for now just do plots for one session
                mat = h5py.File(matfile, 'r')

                in_times, in_trials = event_time_func(mat)
                # TODO: add functionality for markers
                in_nonan_sel = ~np.isnan(in_times)
                in_times_nonan = in_times[in_nonan_sel]

                aligned = [align_activities_with_event(fp_sigs[i], fp_times[i], in_times_nonan,
                                                       time_window, False) for i in range(len(fp_sigs))]

                sigs = ('DA', 'Ca')
                rows = sigs
                cols = ('D1', 'D2')
                meas = 'zscore_' if zscore else '' + 'dF/F'
                aligned_DA, aligned_Ca = aligned[0], aligned[1]

                for i in range(len(rows)):
                    # TODO: add extra event times if needed
                    ax = peristimulus_time_trial_heatmap_plot(aligned[i], time_window, in_trials,
                                                               ("", "time (ms)", ""), ax=axes[i][j])
                    if i == 0:
                        axes[i][j].set_title(cols[j])
                    if j == 0:
                        axes[i][j].set_ylabel(rows[i] + f' ({meas})')
                mat.close()
    zfolder = "zscore" if zscore else "dff"
    # TODO: come up with unifying code for fname
    subfolder = os.path.join(plots, "heatmaps", base_method, zfolder)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    fig.suptitle(f"{tag} times outcome phase")
    fname = os.path.join(subfolder, f"{tag}_heatmap_DACA_D1D2" + "_".join(two_sessions))
    fig.savefig(fname + '.png')
    plt.close("all")


def postUU_switch(animal, session, pre, post, beh_arg, cond_arg, signal, ylims=None, plot_type='trial_avg'):
    event_arg = beh_arg
    fmat0 = drop_empty_nan(giant_eventFP_mat)
    if animal == 'all':
        print('careful about Ca because it is mixed')
    group = {'al': 'mixed_' + region, 'D1': 'D1_' + region, 'A2': 'D2_' + region}[animal[:2]]
    if len(animal) > 3:
        fmat0 = fmat0[fmat0['animal'] == animal].reset_index(drop=True)
    if session != 'all':
        fmat0 = fmat0[fmat0['session'] == session].reset_index(drop=True)
    windowsels = {'DA': (time_window_dict[beh_arg] >= 0) & (time_window_dict[beh_arg] <= 1500),
                  'Ca': (time_window_dict[beh_arg] >= 0) & (time_window_dict[beh_arg] <= 1500)}

    # Post
    # Switch trial: A[t] != A[t-1]
    def plusminus(x):
        if x < 0:
            return str(x)
        elif x == 0:
            return ''
        else:
            return f'+{x}'

    postR = list(range(0, post + 1))  # post switch option
    preR = list(range(pre, 0))  # pre switch option
    rangeS = preR + postR
    nRangeS = [plusminus(x) for x in rangeS]

    postSal = cond_arg.split('_')[1]
    assert pre <= -2, 'Must be at least 2 lags pre'

    if signal == 'all':
        sig_arg = 'DACa'
        allsigs = ['DA', 'Ca']
    else:
        sig_arg = signal
        allsigs = [signal]

    if plot_type.startswith('dimr'):
        dimMethod = plot_type[5:]
        fmat0 = eventFP_mat_neurDimRed(fmat0, dimMethod, allsigs, rangeS, beh_arg, windowsels)

    logsel = (fmat0['A{t-1}'] != fmat0['A']) & np.logical_and.reduce([fmat0['R'] == postSal] + [
        fmat0['R{t+%d}' % n] == postSal for n in range(1, post + 1)]) & np.logical_and.reduce(
        [fmat0['R{t-%d}' % n] == 'U' for n in range(1, abs(pre) + 1)]) & np.logical_and.reduce(
        [fmat0['A{t-1}'] == fmat0['A{t-%d}' % n] for n in range(2, abs(pre) + 1)])
    fmat_plot = fmat0[logsel].reset_index(drop=True)
    all_bins = [f'switch{i}' for i in rangeS]
    fig, axes = plt.subplots(nrows=len(allsigs), ncols=2, sharex=True, sharey='row',
                             figsize=(15, len(allsigs) * 7))
    axes = axes.reshape((len(allsigs), 2))
    plt.subplots_adjust(hspace=0.2, wspace=0.3)

    for i, sig in enumerate(allsigs):
        for j, side in enumerate(['ipsi', 'contra']):
            # Code Mixed switch stay behaviors later
            actsel = np.logical_and.reduce([fmat_plot['A'] == side] +
                                           [fmat_plot['A{t+%d}' % n] == side for n in range(1, post + 1)])
            side_mat = fmat_plot.loc[actsel]
            if side_mat.shape[0] == 0:
                continue
            print(side_mat.shape)
            # Mixed switch stay behaviors TODO: render just stays
            if plot_type.startswith('dimr'):
                dimMethod = plot_type[5:]
                if dimMethod.startswith('pca'):
                    ncomps = int(dimMethod.split('_')[1])
                    template = sig + f'_{event_arg}' + '{t%s}_PC'
                    if ncomps == 1:
                        trials = np.repeat(['t%s' % ll for ll in nRangeS], side_mat.values.shape[0])
                        data = side_mat[[template % ll + '1' for ll in nRangeS]].values.ravel(order='F')
                        pc1arg = sig + f'_{event_arg}_PC1'
                        # add sns.stripplot
                        sns.violinplot(x='trial# periswitch', y=pc1arg,
                                       data=pd.DataFrame({'trial# periswitch': trials, pc1arg: data}),
                                       ax=axes[i, j])

                    elif ncomps == 2:
                        trials = np.repeat(['t%s' % ll for ll in nRangeS], side_mat.values.shape[0])
                        data1 = side_mat[[template % ll + '1' for ll in nRangeS]].values.ravel(order='F')
                        pc1arg = sig + f'_{event_arg}_PC1'
                        data2 = side_mat[[template % ll + '2' for ll in nRangeS]].values.ravel(order='F')
                        pc2arg = sig + f'_{event_arg}_PC2'
                        datapdf = pd.DataFrame({'trial# periswitch': trials,
                                                pc1arg: data1, pc2arg: data2})
                        sns.scatterplot(data=datapdf, x=pc1arg, y=pc2arg, hue='trial# periswitch',  # Joint
                                        ax=axes[i, j])
                    else:
                        raise NotImplementedError(f"Unimplemented with PC {ncomps}")
                elif dimMethod == 'mean':
                    template = sig + f'_{event_arg}' + '{t%s}_mean'
                    columns = [get_columns_FP_df(side_mat, template % ll)[0]
                               for ll in nRangeS]
                    metricN = sig + f'_{event_arg}' + columns[0].split('{')[1].split('}')[1]
                    trials = np.repeat([cc.split('{')[1].split('}')[0] for cc in columns],
                                       side_mat.values.shape[0])
                    data = side_mat[columns].values.ravel(order='F')
                    sns.violinplot(x='trial# periswitch', y=metricN,
                                   data=pd.DataFrame({'trial# periswitch': trials, metricN: data}),
                                   ax=axes[i, j])
            else:
                sub_traces = [None] * len(nRangeS)
                for l, n in enumerate(nRangeS):
                    outcomeI = get_columns_FP_df(side_mat, sig + f'_{event_arg}' + '{t%s}' % n)
                    sub_traces[l] = side_mat[outcomeI].values
                # TODO: Add what post look like, idea: ipsi contra can be mixed as long as it is labeled
                peristimulus_time_trial_average_plot(sub_traces, time_window_dict[beh_arg],
                                                     tags=(None, 'time(ms)', f'{sig} Z(dF/F)', all_bins),
                                                     ax=axes[i, j])
                if ylims is not None:
                    axes[i, j].set_ylim(ylims)
                axes[i, j].legend(fontsize='large')
            if i == 0:
                axes[i, j].set_title(side)
    fig.suptitle(f'{event_arg} {cond_arg} {post} {animal} {session} ' + group)
    switchN = f'pre{pre}_post{post}'
    # plt.savefig(os.path.join(plot_out, f'{sig_arg}_{event_arg}_Periswitch_{plot_type}_{cond_arg}{switchN}_{animal}_{session}.png'))
    # TODO: compare the movement time differences across stay vs switch
    # TODO: different event FP comparison
