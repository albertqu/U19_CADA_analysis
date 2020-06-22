# System
import time, os
# Structure
from collections import deque
# Data
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# caiman
from caiman.source_extraction.cnmf.deconvolution import GetSn


##################################################
#################### Loading #####################
##################################################


def get_sources_from_csv(csvfile):
    pdf = pd.read_csv(csvfile, delimiter=" ", names=['time', 'calcium'], usecols=[0, 1])
    FP_time = pdf.time.values
    FP_signal = pdf.calcium.values

    min_signal, max_signal = np.min(FP_signal), np.max(FP_signal)
    intensity_threshold = min_signal+(max_signal - min_signal)*0.4

    sig470_sels = FP_signal > intensity_threshold
    sig415_sels = ~sig470_sels

    FP_470_signal = FP_signal[sig470_sels]
    FP_470_time = FP_time[sig470_sels]

    FP_415_signal = FP_signal[sig415_sels] #include 470 after unplug
    FP_415_time = FP_time[sig415_sels]
    # TODO: save as pd.DataFrame
    return FP_415_time, FP_415_signal, FP_470_time, FP_470_signal



########################################################
#################### Preprocessing #####################
########################################################


def f0_filter_sig(xs, ys, method=2, width=30):
    """
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(width, method)
    else:
        mf, mDC = std_filter(width, method%10, buffer=True)
    dff = np.array([(mf(ys, i), mDC.get_dev()) for i in range(len(ys))])
    return dff


def calcium_dff(xs, ys, method=2, width=30):
    f0 =f0_filter_sig(xs, ys, method=method, width=width)[:, 0]
    return (ys-f0) / f0


def sources_get_noise_power(s415, s470):
    npower415 = GetSn(s415)
    npower470 = GetSn(s470)
    return npower415, npower470


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


def signal_filter_visualize(FP_415_time, FP_415_signal, FP_470_time, FP_470_signal, samples=200):
    fil = f0_filter_sig(FP_415_time, FP_415_signal, method=12, width=200)[:, 0]
    plt.plot(FP_470_time, FP_470_signal - np.mean(FP_470_signal), 'b-', FP_415_time, FP_415_signal - np.mean(
        FP_415_signal), 'm-')
    plt.plot(FP_415_time, fil - np.mean(FP_415_signal))
    plt.legend(['470', '415',
                'filtered'])


########################################################
#################### Filtering #####################
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
