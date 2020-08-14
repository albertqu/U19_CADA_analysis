from peristimulus import *
from behaviors import *
from utils import get_session_files_FP_ProbSwitch, encode_to_filename, decode_from_filename


#######################################################
################# NAc D1/D2 FP Outcome ################
#######################################################
def NAcD1D2_Fig1_group():
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    choices = {'A2A': {'A2A-15B-B_RT': ["p153"]}, "D1": {"D1-27H_LT": ["p103"]}}
    zscore = False
    base_method = 'robust'

    for group in ['D1', 'A2A']:
        neur_type = group if group == 'D1' else 'D2'
        # hue: reward/unrewarded, row: DA/Ca2+, col: laterality,
        #sessions = get_session_files_FP_ProbSwitch(folder, group, choices=choices, processed=True)
        sessions = choices[group]
        time_window = np.arange(-8000, 2001, 50) # 8sec is max
        for animal in sessions:
            for session in sessions[animal]:
                files = encode_to_filename(folder, animal, session)
                matfile, green, red = files['behavior'], files['green'], files['red']
                # Load FP
                fp_times, fp_sigs, iso_times, iso_sigs = get_sources_from_csvs([green, red],
                                                                            tags=('DA', 'Ca'), show=False)

                fp_sigs = [raw_fluor_to_dff(fp_times[i], fp_sigs[i], iso_times[i], iso_sigs[i], base_method,
                                           zscore=zscore) for i in range(len(fp_sigs))]
                # TODO: for now just do plots for one session
                mat = h5py.File(matfile, 'r')
                # TODO: investigate into usage of seaborn hue, row, col function for plotting
                outcome_times = get_outcome_times(mat)
                outcome_nonan_sel = ~np.isnan(outcome_times)
                outcome_times_nonan = outcome_times[outcome_nonan_sel]
                trial_outcomes = get_trial_outcomes(mat)
                trial_laterality = get_trial_outcome_laterality(mat)
                # handle the nan in outcome times
                aligned = [align_activities_with_event(fp_sigs[i], fp_times[i], outcome_times_nonan,
                                                       time_window, False) for i in range(len(fp_sigs))]
                aligned_DA, aligned_Ca = aligned[0], aligned[1]
                # TODO: trial_by_trial_time_lock to investigate time warp with choice_time
                sigs = ('DA', 'Ca')
                rows = sigs
                cols = ('ipsi', 'contra')
                hues = ['Incorrect', 'Correct Omission', 'Rewarded']
                meas = 'zscore_' if zscore else '' + 'dF/F'
                fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey='row',
                                         figsize=(20, 10))
                for i in range(len(axes)):
                    # TODO: add extra event times if needed
                    for j in range(len(axes[i])):
                        csel =trial_laterality[cols[j]][outcome_nonan_sel]
                        ax = peristimulus_time_trial_average_plot([aligned[i][trial_outcomes[h]
                                                                              [outcome_nonan_sel] & csel]
                                                                   for h in hues], time_window,
                                                                  ("outcome", "time(ms)", "", hues),
                                                                  ax=axes[i][j])
                        if i == 0:
                            axes[i][j].set_title(cols[j])
                    axes[i][0].set_ylabel(rows[i] + f' ({meas})')
                fig.suptitle(f"{neur_type} outcome phase DA, Ca")

                zfolder = "zscore" if zscore else "dff"
                # TODO: come up with unifying code for fname
                subfolder = os.path.join(plots, f"{animal}", base_method, zfolder)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                fname = os.path.join(subfolder, f"{neur_type}_outcome_DACA_laterality_result_{session}")
                fig.savefig(fname + '.png')
                plt.close("all")
                mat.close()


# TODO: in need of function (event_time, row: [booleans], col: boolean, hue: booleans, plot_type) -> proper
#  2x2 plots
def NAcD1D2_Fig2_group():
    # hue: ITI, row: trial history, col: laterality
    show = False
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    choices = {'A2A': {'A2A-15B-B_RT': ["p153"]}, "D1": {"D1-27H_LT": ["p103"]}}
    zscore = False # Should not matter with 1 session
    base_method = 'robust'

    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        neur_type = group if group == 'D1' else 'D2'
        sessions = choices[group]
        time_window = np.arange(-8000, 2001, 50)
        for animal in sessions:
            for session in sessions[animal]:
                files = encode_to_filename(folder, animal, session)
                matfile, green, red = files['behavior'], files['green'], files['red']
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

                # TODO: investigate into usage of seaborn hue, row, col function for plotting
                outcome_times = get_outcome_times(mat)
                outcome_nonan_sel = ~np.isnan(outcome_times)
                outcome_times_nonan = outcome_times[outcome_nonan_sel]

                itis = access_mat_with_path(mat, "glml/trials/ITI", ravel=True)
                intervals = [(1.05, 4), (0.65, 1.05), (0.5, 0.65), (0, 0.5)]
                itis_binned = {str(itvl): (itis > itvl[0]) & (itis <= itvl[1]) for itvl in intervals}
                trial_laterality = get_trial_outcome_laterality(mat)
                trial_outcomes = get_trial_outcomes(mat, as_array=True)
                past1_outcome = ['Incorrect', 'Correct Omission', 'Rewarded']
                outcome_code = {'No choice': 3, 'Incorrect': 2, 'Correct Omission': 1.1, 'Rewarded': 1.2}
                trial_history = np.full_like(trial_outcomes, 3)
                trial_history[1:] = trial_outcomes[:-1]
                thist_binned = {"t-1="+po[0]: trial_history == outcome_code[po] for po in past1_outcome}

                aligned = [align_activities_with_event(fp_sigs[i], fp_times[i], outcome_times_nonan,
                                                       time_window, False) for i in range(len(fp_sigs))]
                aligned_DA, aligned_Ca = aligned[0], aligned[1]
                sigs = ('DA', 'Ca')
                rows = ('t-1=I', 't-1=C', 't-1=R')
                cols = ('ipsi', 'contra')
                meas = 'zscore_' if zscore else '' + 'dF/F'
                hues = [str(itvl) for itvl in intervals]

                zfolder = "zscore" if zscore else "dff"
                # TODO: come up with unifying code for fname
                subfolder = os.path.join(plots, f"{animal}", base_method, zfolder)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)

                for k, fsig in enumerate(sigs):
                    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey='row',
                                             figsize=(20, 10))
                    for i in range(len(axes)):
                        # TODO: add extra event times if needed
                        axes[i][0].set_ytitle = rows[i] + f' ({meas})'
                        for j in range(len(axes[i])):
                            rsels = thist_binned[rows[i]][outcome_nonan_sel]
                            csels = trial_laterality[cols[j]][outcome_nonan_sel]
                            ax = peristimulus_time_trial_average_plot([aligned[k][itis_binned[h]
                                                                     [outcome_nonan_sel] & rsels & csels]
                                                                        for h in hues],
                                                                      time_window,
                                                                      ("outcome", "time(ms)", "", hues),
                                                                      ax=axes[i][j])
                            if i == 0:
                                axes[i][j].set_title(cols[j])
                        axes[i][0].set_ylabel(rows[i] + f' ({meas})')
                    fig.suptitle(f"trial history & ITI effects on {neur_type} outcome phase {fsig}")
                    fname = os.path.join(subfolder,f"{neur_type}_{fsig}_outcome_trialH_side_ITIbin_{session}")
                    fig.savefig(fname + '.png')
                    plt.close("all")
                mat.close()


def NAcD1D2_Fig3_group():
    # TODO: add functionality for markers

    def center_in_func(mat):
        center_in_times = access_mat_with_path(mat, 'glml/time/center_in', ravel=True)
        center_in_trials = access_mat_with_path(mat, 'glml/trials/center_in', ravel=True,
                                                dtype=np.int)
        return center_in_times, center_in_trials

    trial_heatmap_D1D2(center_in_func, "center_in")


def outcome_func(mat):
    outcome_times = get_outcome_times(mat)
    outcome_trials = np.where(~np.isnan(outcome_times))[0]
    return outcome_times, outcome_trials


def trial_heatmap_D1D2(event_time_func, tag):
    show = False
    folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
    choices = {'A2A': {'A2A-15B-B_RT': ["p153"]}, "D1": {"D1-27H_LT": ["p103"]}}
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
                matfile, green, red = files['behavior'], files['green'], files['red']
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

