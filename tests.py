from behaviors import *
from peristimulus import *


def test_FP_preprocessing():
    show = True
    folder_load = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_Raw"
    folder_save = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    animal, session = 'A2A-15B-B_RT', 'p153_FP_LH'
    files = encode_to_filename(folder_load, animal, session)
    files2 = encode_to_filename(folder_save, animal, session, ['behavior', 'FP'])
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

    zscore = False
    behaviors = ['outcome']
    time_window = np.arange(-2000, 2001, 50)
    mat = h5py.File(matfile, 'r')
    # Get aligned signals to behaviors
    # default to align first
    behavior_times = np.vstack([get_behavior_times(mat, beh) for beh in behaviors])
    nonan_sel = ~np.any(np.isnan(behavior_times), axis=0)
    behavior_times_nonan = behavior_times[:, nonan_sel]
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
