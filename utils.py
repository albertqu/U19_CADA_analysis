# System
import time, os, h5py, re
# Structure
from collections import deque
# Data
import scipy
import numpy as np
import pandas as pd
from scipy.sparse import diags as spdiags
from scipy.sparse import linalg as sp_linalg
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# caiman
try:
    from caiman.source_extraction.cnmf.deconvolution import GetSn
    from caiman.source_extraction.cnmf.utilities import fast_prct_filt
    from caiman.utils.stats import df_percentile
except ModuleNotFoundError:
    print("CaImAn not installed or environment not activated, certain functions might not be usable")


# TODO: Move project specific portions to pipeline_*.py as things scale

##################################################
#################### Loading #####################
##################################################


def get_session_files_FP_ProbSwitch(folder, group, photometry='both', choices=None, processed=True):
    """ Returns lists of session files of different recording type
    :param group: str, expression
    :param photometry:
    :param choices:
    :param processed:
    :return:
    """
    only_Ca = []
    only_DA = []
    if processed:
        results = []
        return
    else:
        matfiles, red, green, binary, timestamps = [], [], [], [], []
        return matfiles, red, green, binary, timestamps


def get_prob_switch_all_sessions(folder):
    """ Exhaustively check all folder that contains ProbSwitch task .mat files and encode all sessions.
    .mat -> decode -> return group
    :param folder:
    :return:
    """
    pass


def get_sources_from_csv(csvfile, window = 400):
    pdf = pd.read_csv(csvfile, delimiter=" ", names=['time', 'calcium'], usecols=[0, 1])
    FP_time = pdf.time.values
    FP_signal = pdf.calcium.values

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
    down_dyn_std = np.std(FP_down_signal[:downN * window].reshape((downN, window), order='C'), axis=1).mean()
    # TODO: check for consecutives
    # TODO: check edge case when only 415 has signal
    if top_dyn_std >= down_dyn_std:
        sigREC_sel, sig415_sel = sigT_sels, sigD_sels
        FP_REC_signal, FP_REC_time = FP_top_signal, FP_top_time
        FP_415_signal, FP_415_time = FP_down_signal, FP_down_time
    else:
        sigREC_sel, sig415_sel = sigD_sels, sigT_sels
        FP_REC_signal, FP_REC_time = FP_down_signal, FP_down_time
        FP_415_signal, FP_415_time = FP_top_signal, FP_top_time

    # TODO: visualization of the separated channel, figure out best color

    # TODO: save as pd.DataFrame
    return FP_415_time, FP_415_signal, FP_REC_time, FP_REC_signal


def decode_from_filename(filename):
    """
    `A2A-15B_RT_20200612_ProbSwitch_p243_FP_RH`, `D1-27H_LT_20200314_ProbSwitch_FP_RH_p103`
    behavioral: * **Gen-ID_EarPoke_Time_DNAME_Age_special.mat**
FP: **Gen-ID_EarPoke_DNAME2_Hemi_Age_channel_Time(dash)[Otherthing].csv**
binary matrix: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_Binary_Matrix)Time[special].etwas**
timestamps: **Drug-ID_Earpoke_DNAME_Hemi_Age_(NIDAQ_Ai0_timestamps)Time[special].csv**
    GEN: genetic line, ID: animal ID, EP: ear poke, T: time of expr, TD: detailed HMS DN: Data Name, A: Age,
    H: hemisphere, S: session, SP: special extension
    :param filename:
    :return:
    """
    # case behavior
    mBMat = re.search(r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<T>\d+)_(?P<DN>[-&\w]+)_("
                      r"?P<A>p\d+)(?P<SP>[-&\w]*)\.mat", filename)
    # case binary
    mBIN = None
    options, ftype = None, None
    if mBMat is not None:
        options = mBMat.groupdict()
        ftype = "behavior"
    elif mBIN is not None:
        options = mBIN.groupdict()
        ftype = "bin_mat"
    else:
        # case csv
        #todo: merge cage id and earpoke
        """A2A-16B-1_RT_ChR2_switch_no_cue_LH_p147_red_2020-03-17T15_38_40.csv"""
        channels = ['keystrokes', "MetaData", "NIDAQ_Ai0_timestamp", "red", "green"]
        for c in channels:
            mCSV = re.search(
                r"^(?P<GEN>\w{2,3})-(?P<ID>\d{2,}[-\w*]*)_(?P<EP>[A-Z]{2})_(?P<DN>[-&\w]+)_(?P<H>[LR]H)_"
                r"(?P<A>p\d+)(?P<SP>[-&\w]*)" + f"_{c}" + r"(?P<S>_session\d+_|_?)(?P<T>\d{4}-?\d{2}-?\d{2})T"
                r"(?P<TD>[_\d]+)\.csv", filename)

            if mCSV is not None:
                options = mCSV.groupdict()
                # if options['CH'] == 'green' or options['CH'] == 'red':
                #     ftype = "FP"
                # if "session2" in filename:
                ftype = c
                # print(filename)
                # print(options)
        if ftype is None:
            print("special:", filename)


def encode_to_filename(options, ftype):
    """ TODO: refer to caiman for function
    :param options:
    :param ftype: if ftype=="all": returns all 5 files
    :return:
    """
    pass


def access_mat_with_path(mat, p, ravel=False, dtype=None, raw=False):
    """ Takes in .mat file or hdf5 file and path like structure p and return the entry
    :param mat:
    glml matfiles: modified from Belief_State_FP_Analysis.m legacy Chris Hall GLM structure:
        glml/
            notes/
                hemisphere/
                region/
            time/
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
            trials/ (1-indexed)
                ITI/
                center_to_side/
                contra/
                contra_rew/
                contra_unrew/
                execute/
                initiate/
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
                execute/
                initiate/
                ipsi/
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


########################################################
#################### Preprocessing #####################
########################################################


def sources_get_noise_power(s415, s470):
    npower415 = GetSn(s415)
    npower470 = GetSn(s470)
    return npower415, npower470


def f0_filter_sig(xs, ys, method=2, window=200):
    """
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=True)
    dff = np.array([(mf(ys, i), mDC.get_dev()) for i in range(len(ys))])
    return dff


def percentile_filter(xs, ys, window=200, perc=None):
    # TODO: 1D signal only
    if perc is None:
        perc, val = df_percentile(ys[:window])
    return scipy.ndimage.percentile_filter(ys, perc, window)


def isosbestic_baseline_correct(xs, ys, method=12, window=200):
    # TODO: this is the greedy method with only the mean estimation
    return f0_filter_sig(xs, ys, method=method, window=window)[:, 0]


def calcium_dff(xs, ys, xs0=None, y0=None, method=2, window=200):
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
                            isosbestic=True, window=200):
    # For visualize purpose, all signals are demeaned first:
    # TODO: add exclude event property
    FP_470_signal = FP_470_signal - np.mean(FP_470_signal)
    FP_415_signal = FP_415_signal - np.mean(FP_415_signal)
    if isosbestic:
        f0 = isosbestic_baseline_correct(FP_415_time, FP_415_signal, window=window)
        n415, n470 = sources_get_noise_power(FP_415_signal, FP_470_signal)
        std415, std470 = np.std(FP_415_signal, ddof=1), np.std(FP_470_signal, ddof=1)
        f0_npower_correct = f0 * n470 / n415
        f0_std_correct = f0 * std470 / std415
        plt.plot(FP_470_time, FP_470_signal, 'b-')
        plt.plot(FP_415_time, FP_415_signal, 'm-')
        plt.plot(FP_415_time, np.vstack([f0, f0_npower_correct, f0_std_correct]).T)
        plt.legend(['470 channel', '415 channel (isosbestic)', 'raw baseline', 'noise-power-correct',
                    'sig-power-correct'])
    else:
        f0_rstd = f0_filter_sig(FP_470_time, FP_470_signal, method=12, window=window)[:, 0]
        # similar to Pnevmatikakis 2016 and caiman library
        f0_perc15 = percentile_filter(FP_470_time, FP_470_signal, window=window, perc=15)
        f0_percAuto = percentile_filter(FP_470_time, FP_470_signal, window=window, perc=None)
        plt.plot(FP_470_time, FP_470_signal, 'b-')
        plt.plot(FP_415_time, FP_415_signal, 'm-')
        plt.plot(FP_470_time, np.vstack([f0_perc15, f0_percAuto, f0_rstd]).T)
        plt.legend(['470 channel', '415 channel', '15-percentile', 'mode-percentile', 'robust-std-filter'])
    plt.xlabel('frames')
    plt.ylabel('Fluorescence (demeaned)')


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


########################################################
###################### Filtering #######################
########################################################
class DCache:
    # TODO: AUGMENT IT SUCH THAT IT WORKS FOR MULTIPLE

    def __init__(self, size=20, thres=2, buffer=False, ftype='mean'):
        """
        :param size: int, size of the dampening cache
        :param thres: float, threshold for valid data caching, ignore signal if |x - mu_x| > thres * var
        :param buffer: boolean, for whether keeping a dynamic buffer
        so far cache buffer only accepts 1d input
        """
        self.size = size
        self.thres = thres
        self.counter = 0
        self.bandwidth = None
        self.ftype = ftype
        if ftype == 'median':
            assert buffer, 'median filter requires buffer'
        else:
            assert ftype == 'mean', 'filter type undefined'

        if buffer:
            self.cache = deque()
            self.avg = 0
            self.dev = 0
        else:
            self.cache = None
            self.avg = 0
            self.m2 = 0
            self.dev = 0

    def __len__(self):
        return self.size

    def update_model(self):
        if self.ftype == 'median':
            self.avg = np.nanmedian(self.cache)
            self.dev = np.median(np.abs(np.array(self.cache) - self.avg))
        elif self.cache is not None:
            self.avg = np.nanmean(self.cache)
            self.dev = np.std(self.cache)
        else:
            self.dev = np.sqrt(self.m2 - self.avg ** 2)

    def add(self, signal):
        # handle nans:
        if self.cache is not None:
            assert np.prod(np.array(signal).shape) == 1, 'cache buffer only supports scalar so far'
            if not np.isnan(signal):
                if self.counter < self.size:
                    self.cache.append(signal)
                else:
                    if (signal - self.avg) < self.get_dev() * self.thres:
                        self.cache.append(signal)
                        self.cache.popleft()
                self.counter += 1
        else:
            if self.bandwidth is None:
                self.bandwidth = signal.shape[0]
            if self.counter < self.size:
                if np.sum(np.isnan(signal)) > 0:
                    #print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
                    self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
                    self.m2 = (signal ** 2 + self.m2 * self.counter) / (self.counter+1)
                    self.counter += 1
            else:
                targets = (~np.isnan(signal)) & ((signal - self.avg) < self.get_dev() * self.thres)
                #print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
                self.avg[targets] = (self.avg[targets] * (self.size - 1) + signal[targets]) / self.size
                self.m2[targets] = (signal[targets] ** 2 + self.m2[targets] * (self.size - 1)) / self.size
                self.counter += 1
        self.update_model()

    def get_val(self):
        return self.avg

    def get_dev(self):
        return self.dev


def std_filter(width=20, s=2, buffer=False):
    dc = DCache(width, s, buffer=buffer)

    def fil(sigs, i):
        dc.add(sigs[i])
        #print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc


def median_filter(width=20, s=2):
    dc = DCache(width, s, buffer=True, ftype='median')

    def fil(sigs, i):
        dc.add(sigs[i])
        # print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc


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
    TODO: implement more detailed progress with subtasks
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
