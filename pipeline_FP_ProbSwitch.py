from peristimulus import *
from behaviors import *
from utils import get_session_files_FP_ProbSwitch, encode_to_filename, decode_from_filename


#######################################################
################# NAc D1/D2 FP Outcome ################
#######################################################
def NAcD1D2_Fig1_group():
    folder = ""
    choices = None

    # choices = {'D1': [], 'A2A': []}
    for group in ['D1', 'A2A']:
        neur_type = group if group == 'D1' else 'D2'
        # hue: reward/unrewarded, row: DA/Ca2+, col: laterality,
        sessions = get_session_files_FP_ProbSwitch(folder, group, choices=choices, processed=True)
        time_window = np.arange(-8000, 2001, 50) # 8sec is max
        for animal in sessions:
            for session in sessions[animal]:
                matfile, red, green = encode_to_filename(animal, session)

                # Load FP
                # TODO: for now just do plots for one session
                mat = h5py.File(matfile, 'r')
                fp_sigs, fp_times = None, None
                # neural_sigs, neural_times (2 x T)
                # TODO: investigate into usage of seaborn hue, row, col function for plotting
                # {row:, col: , hue: }
                #  align_activities_with_event(sigs, times, event_times, time_window, discrete=True)
                outcome_times = get_outcome_times(mat)
                trial_outcomes = get_trial_outcomes(mat)
                trial_laterality = get_trial_outcome_laterality(mat)

                aligned = align_activities_with_event(fp_sigs, fp_times, outcome_times, time_window, False)
                aligned_DA, aligned_Ca = aligned[0], aligned[1]
                sigs = ('DA', 'Ca')
                cols = ('ipsi', 'contra')
                hues = ('rewarded', 'unrewarded')
                fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
                for i in range(len(axes)):
                    # TODO: add extra event times if needed
                    # peristimulus_time_trial_average_plot(sigs, times, tags, extra_event_times=None, ax=None)
                    for j in range(len(axes[i])):
                        ax = peristimulus_time_trial_average_plot([aligned[trial_outcomes[h] &
                                                                           trial_laterality[cols[j]]]
                                                                            for h in hues],
                                                                  time_window, hues, ax=axes[i][j])
                fig.suptitle(f"{neur_type} outcome phase DA, Ca2+")
                mat.close()


# TODO: in need of function (event_time, row: [booleans], col: boolean, hue: booleans, plot_type) -> proper
#  2x2 plots
def NAcD1D2_Fig2_group():
    # hue: ITI, row: trial history, col: laterality
    folder = ""
    choices = None

    # choices = {'D1': [], 'A2A': []}
    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        # hue: reward/unrewarded, col: laterality,
        neur_type = group if group == 'D1' else 'D2'
        sessions = get_session_files_FP_ProbSwitch(folder, group, choices=choices, processed=True)
        time_window = np.arange(-8000, 2001, 50)
        for animal in sessions:
            for session in sessions[animal]:
                matfile, red, green = encode_to_filename(animal, session)

                # Load FP
                # TODO: for now just do plots for one session
                mat = h5py.File(matfile, 'r')
                fp_sigs, fp_times = None, None
                # neural_sigs, neural_times (2 x T)
                # TODO: investigate into usage of seaborn hue, row, col function for plotting
                # {row:, col: , hue: }
                #  align_activities_with_event(sigs, times, event_times, time_window, discrete=True)
                outcome_times = get_outcome_times(mat)
                itis = access_mat_with_path(mat, "glml/trials/ITIs", ravel=True)
                intervals = [(0, 0.5), (0.5, 0.65), (0.65, 1.05), (1.05, 4)]
                itis_binned = {str(itvl): (itis > itvl[0]) & (itis <= itvl[1]) for itvl in intervals}
                trial_laterality = get_trial_outcome_laterality(mat)
                trial_outcomes = get_trial_outcomes(mat)
                past1_outcome = ['Unrewarded', 'Correct Omission', 'Rewarded']
                outcome_code = {'N/A': 0, 'Unrewarded': 1.1, 'Correct Omission': 1.2, 'Rewarded': 2}
                trial_history = np.zeros_like(trial_outcomes)
                trial_history[1:] = trial_outcomes[:-1]
                thist_binned = {"t-1="+po[0]: trial_history==outcome_code[po] for po in past1_outcome}

                aligned = align_activities_with_event(fp_sigs, fp_times, outcome_times, time_window, False)
                aligned_DA, aligned_Ca = aligned[0], aligned[1]
                sigs = ('DA', 'Ca')
                rows = ('t-1=U', 't-1=C', 't-1=R')
                cols = ('ipsi', 'contra')
                hues = intervals
                for k, fsig in enumerate(sigs):
                    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
                    for i in range(len(axes)):
                        # TODO: add extra event times if needed
                        # peristimulus_time_trial_average_plot(sigs, times, tags, extra_event_times=None, ax=None)
                        for j in range(len(axes[i])):
                            rsels = thist_binned[rows[i]]
                            csels = trial_laterality[cols[j]]
                            ax = peristimulus_time_trial_average_plot([aligned[k, rsels & csels &
                                                                               itis_binned[h], :]
                                                                        for h in hues],
                                                                      time_window, hues, ax=axes[i][j])
                    fig.suptitle(f"trial history & ITI effects on {neur_type} outcome phase {sigs[k]}")
                mat.close()


def NAcD1D2_Fig3_group():
    # heatmap: row: D1, col: D2
    folder = ""
    choices = None

    # choices = {'D1': [], 'A2A': []}
    # hue: ITI, row,  col: laterality
    for group in ['D1', 'A2A']:
        # hue: reward/unrewarded, col: laterality,
        sessions = get_session_files_FP_ProbSwitch(folder, group, choices=choices, processed=True)
        time_window = np.arange(-2000, 2001, 50)
        for animal in sessions:
            for session in sessions[animal]:
                matfile, red, green = encode_to_filename(animal, session)

                # Load FP
                # TODO: for now just do plots for one session
                # TODO: add center in, and center times for all files

                aligned = time_aligned_from_files()