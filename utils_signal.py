# Structure
from collections import deque
# Data
import scipy
import numpy as np
RAND_STATE = 230

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

    def set_init(self, avg, m2):
        self.avg = avg
        self.m2 = m2
        self.dev = np.sqrt(self.m2 - self.avg ** 2)
        # TODO: figure out more formal way
        self.counter = 1

    def add(self, signal):
        # handle nans:
        if np.issubdtype(signal, np.number):
            signal = np.array([signal])
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
                if len(signal.shape) == 0:
                    self.bandwidth = 1
                else:
                    self.bandwidth = signal.shape[0]
            if self.counter < self.size:
                if np.sum(~np.isnan(signal)) > 0:
                    #print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
                    self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
                    self.m2 = (signal ** 2 + self.m2 * self.counter) / (self.counter+1)
                    self.counter += 1
            else:
                # TODO: make two-sided
                targets = (~np.isnan(signal)) & ((signal - self.avg) < self.get_dev() * self.thres)
                #print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
                self.avg[targets] = (self.avg[targets] * (self.size - 1) + signal[targets]) / self.size
                self.m2[targets] = (signal[targets] ** 2 + self.m2[targets] * (self.size - 1)) / self.size
                self.counter += 1
        self.update_model()

    def get_val(self):
        # avg has to be vector
        if isinstance(self.avg, np.ndarray) and len(self.avg) == 1:
            return self.avg[0]
        return self.avg

    def get_dev(self):
        if isinstance(self.dev, np.ndarray) and len(self.dev) == 1:
            return self.dev[0]
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


def robust_filter(ys, method=12, window=200, optimize_window=2, buffer=False):
    """
    First 2 * windows re-estimate with mode filter
    To avoid edge effects as beginning, it uses mode filter; better solution: specify initial conditions
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=buffer)
    opt_w = int(np.rint(optimize_window * window))
    # prepend
    init_win_ys = ys[:opt_w]
    prepend_ys = init_win_ys[opt_w-1:0:-1]
    ys_pp = np.concatenate([prepend_ys, ys])
    f0 = np.array([(mf(ys_pp, i), mDC.get_dev()) for i in range(len(ys_pp))])[opt_w-1:]
    return f0
