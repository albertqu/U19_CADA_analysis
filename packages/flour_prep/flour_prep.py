import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pywt

from sklearn.linear_model import Lasso, LassoCV

''' Photometry Pre-Processing Class '''
class Preprocess:
    '''
        @author  Cameron Jordan
        @email   cameronjordan@berkeley.edu
        @version 1.0
    '''

    def __init__(self, timeseries, signal, reference, positive_coefficients=False, sampling_frequency=20, drop=200, window_size=11, r_squared_threshold=0.7):
        '''
        Initialize a Preprocess Object

            Parameters:
                timeseries            : Sequence of Data Points Over a Time Variable
                signal                : Values of the Signal Over the Above timeseries
                reference             : Values of the Reference Signal Over the Above timeseries
                positive_coefficients : Indicates Whether the LASSO Regression Can Have Positive Coefficients; boolean
                sampling_frequency    : Sampling Frequency of the Equipment
                drop                  : Number of Initial Frames to Drop
                window_size           : Desired Window Size for Triangular Moving Average Smoothing
                r_squared_threshold   : r_squared Cutoff Value for Baseline Similarity
        '''

        assert window_size % 2 == 1, "the window size must be odd"

        self.ts = timeseries[drop:]
        self.signal = signal[drop:]
        self.smoothed_signal = None
        self.detrended_signal = None
        self.z_dFF = None
        self.ref = reference[drop:]
        self.smoothed_ref = None
        self.detrended_ref = None
        self.fitted_ref = None
        self.detrended = False
        self.positive_coefficients = positive_coefficients
        self.fs = sampling_frequency
        self.drop = drop
        self.window_size = window_size
        self.r_squared_threshold = r_squared_threshold

    def pipeline(self, smoothing_method, baseline_method, fit_method, detrend_last=False, show=False, ax=None):
        '''
        Constructs a Pre-Processing Pipeline

            Parameters:
                smoothing_method : Method to Pass to self.smooth(); string-type
                    - tma [triangular_moving_average]
                    - ss  [smoothing_spline]
                baseline_method  : Method to Pass to self.compare_and_baseline() and self.baseline() (if detrend_last == True); string-type
                    - w   [wavelet]
                    - lpf [low_pass_filter]
                fit_method       : Method to Pass to self.fit(); string-type
                    - l    [lasso] 
                detrend_last     : Indicates Whether to Detrend After Subtracting self.fitted_ref From self.detrended_signal
                ax               : Axis to Pass to _visualize; plt.gca() object
                show             : Indicates Whether to Graph the Pre-Processed Signal Upon Pipeline Generation

            Returns:
                self.z_dFF       : Normalized, Filtered, Baseline-Corrected signal Channel

        '''

        self.smooth(smoothing_method)
        self.compare_and_baseline(baseline_method)
        self.fit(fit_method)

        if (self.detrended):
            self.z_dFF = (self.detrended_signal - self.fitted_ref)
        else:
            self.z_dFF = (self.detrended_signal - self.fitted_ref) / self.fitted_ref

        if (detrend_last):
            self.z_dFF = self.baseline(baseline_method, self.z_dFF)

        if (show):
            self._visualize(self.z_dFF, 'z_dFF', ax=ax)
        
        return self.z_dFF

    def smooth(self, method):
        '''
        Apply Selected Smoothing Function to Signal

            Parameters:
                method : string-type
                    - tma [triangular_moving_average]
                    - ss  [smoothing_spline]

            Returns:
                self.smoothed_signal : array-like
                self.smoothed_ref : array-like

        '''

        if (method == "tma" or method == "triangular_moving_average"):
            self.smoothed_signal = self.triangular_moving_average(self.signal)
            self.smoothed_ref = self.triangular_moving_average(self.ref)
        elif (method == "ss" or method == "smoothing_spline"):
            self.smoothed_signal = sp.interpolate.make_smoothing_spline(self.ts, self.signal)(self.ts)
            self.smoothed_ref = sp.interpolate.make_smoothing_spline(self.ts, self.ref)(self.ts)
        else:
            raise Exception("Unsupported Smoothing Method")

        return self.smoothed_signal, self.smoothed_ref

    def compare_and_baseline(self, method):
        '''
        Apply Selected Baseline Determination Function to Signal

        * Determines Baseline Signal using Selected Determination Function
        * If signal_baseline Significantly Differs From reference_baseline
            * Detrend Both self.signal and self.reference
            * Return detrended_signal and detrended_reference
        * Else
            * Return self.signal and self.reference

            Parameters:
                method : string-type
                    - w   [wavelet]
                    - lpf [low_pass_filter]

            Returns:
                self.detrended_signal : array-like
                self.detrended_ref : array_like

        '''

        if (method == "wavelet" or method == "w"):
            signal_baseline = self.wavelet_baseline(self.smoothed_signal)
            reference_baseline = self.wavelet_baseline(self.smoothed_ref)
        elif (method == "low_pass_filter" or method == "lpf"):
            signal_baseline = self.lpf_baseline(self.smoothed_signal)
            reference_baseline = self.lpf_baseline(self.smoothed_ref)
        else:
            raise Exception("Unsupported Detrending Method")
        self.signal_baseline = signal_baseline
        self.reference_baseline = reference_baseline

        Y = signal_baseline[self.drop:]
        X = sm.add_constant(reference_baseline[self.drop:])
        model = sm.OLS(Y,X)
        results = model.fit()
        r_squared = results.rsquared
        self.r2 = r_squared

        if (r_squared < self.r_squared_threshold):
            self.detrended_signal = self.smoothed_signal - signal_baseline[:len(self.smoothed_signal)]
            self.detrended_ref = self.smoothed_ref # - reference_baseline[:len(self.smoothed_ref)]
            self.detrended = True
        else:
            self.detrended_signal = self.smoothed_signal
            self.detrended_ref = self.smoothed_ref

        return self.detrended_signal, self.detrended_ref

    def baseline(self, method, signal):
        '''
        Apply Selected Baseline Determination Function to Signal

        * Determines Baseline Signal using Selected Determination Function
        * Detrend Both self.signal and self.reference

            Parameters:
                method : string-type
                    - w   [wavelet]
                    - lpf [low_pass_filter]

            Returns:
                detrended_signal : array-like

        '''

        if (method == "wavelet" or method == "w"):
            signal_baseline = self.wavelet_baseline(signal)
        elif (method == "low_pass_filter" or method == "lpf"):
            signal_baseline = self.lpf_baseline(signal)
        else:
            raise Exception("Unsupported Detrending Method")

        detrended_signal = signal - signal_baseline

        return detrended_signal

    def fit(self, method):
        '''
        Fit the Reference Signal to the Signal

            Parameters:
                method : string-type
                    - l    [lasso]

            Returns:
                self.z_fitted_ref : array-like

        '''

        self.z_signal = self.detrended_signal
        self.z_ref = self.detrended_ref

        if (method == "lasso" or method == "l"):
            if (self.positive_coefficients):
                lin = Lasso(alpha=0,precompute=True,max_iter=1000, positive=True, random_state=9999, selection='random')
            else:
                lin = Lasso(alpha=0,precompute=True,max_iter=1000, positive=False, random_state=9999, selection='random')
            lin.fit(self.detrended_ref.reshape(len(self.detrended_ref),1), self.detrended_signal.reshape(len(self.detrended_ref),1))
            print(lin.coef_)
            self.fitted_ref = lin.predict(self.detrended_ref.reshape(len(self.detrended_ref),1)).reshape(len(self.detrended_ref),)

        return self.fitted_ref

    #######################
    ## Smoothing Methods ##
    #######################

    def triangular_moving_average(self, signal):
        '''
        Applies a Triangular Moving Average to signal

            Parameters:
                signal : Signal to be Smoothed; array-like

            Returns:
                smoothed_signal : Triangular Moving Average Smoothed Signal; array-like

        '''

        window = np.concatenate((np.arange(1, (self.window_size + 1) // 2 + 1), np.arange((self.window_size + 1) // 2 - 1, 0, -1)))
        window = window / window.sum()
        padded_data = np.pad(signal, (self.window_size // 2, self.window_size // 2), mode='edge')
        smoothed_signal = np.convolve(padded_data, window, mode='valid')
        return smoothed_signal

    ######################
    ## Baseline Methods ##
    ######################

    def wavelet_baseline(self, signal, wavelet='sym8'):
        '''
        Applies a Wavelet Decomposition (to the lowest frequency component), Then Reconstructs the Baseline Component

            Parameters:
                signal  : Signal Whose Baseline is to be Determined; array-like
                wavelet : Wavelet to Use in Decomposition/Reconstruction; string-type

            Returns:
                baseline : Signal Baseline (as determined by wavelet methods); array-like

        '''

        level = pywt.dwt_max_level(len(signal), wavelet)
        coeffs = pywt.wavedec(signal, wavelet=wavelet, mode='periodization', level=level)

        for i in range (1, level + 1):
            coeffs[-i] = np.zeros_like(coeffs[-i])

        baseline = pywt.waverec(coeffs, wavelet=wavelet, mode='periodization')
        return baseline

    def lpf_baseline(self, signal, cutoff=(1/600)):
        '''
        Applies a Low-Pass Filter to the Signal to Determine the Low Frequency (Baseline) Components

            Parameters:
                signal : Signal Whose Baseline is to be Determined; array-like
                cutoff : LPF Cut-Off Frequency; float

            Returns:
                baseline : Signal Baseline (as determined by lpf); array-like

        '''

        sos = sp.signal.butter(4, cutoff, 'low', fs=self.fs, output='sos')
        baseline = sp.signal.sosfiltfilt(sos, signal)
        return baseline

    #####################
    ## Utility Methods ##
    #####################

    def _z_score(self, signal):
        '''
        Calculate the Z Score 

            Parameters:
                signal : Signal to z-Score; array-like

            Returns:
                z_signal : z-Scored Signal
                
        '''

        z_signal = (signal - np.median(signal)) / np.std(signal)

        return z_signal

    def _visualize(self, signal, title, ax=None, **kwargs):
        '''
        Visualize the Specified Signal using matplotlib

            Parameters:
                signal : The Signal Data to be Visualized; array-like
                title  : Title of the Plot; array-like, string-type
                ax     : Axis to Plot signal On; plt.gca() object

        '''

        if ax is None:
            hold = False;
            plt.figure(figsize=(30, 5))
            ax = plt.gca()
        else:
            hold = True

        ax.plot(self.ts, signal, label=title, **kwargs)
        plt.legend()
        sns.despine()

        if not hold:
            sns.despine()
            plt.show()

    def _visualize_psd(self, signal, title, ax=None, **kwargs):
        '''
        Visualize the Power Spectral Density of the Specified Signal using matplotlib

            Parameters:
                signal : The Signal Whose Power Spectrum is to be Visualized; array-like
                title  : Title of the Plot; array-like, string-type
                ax     : Axis to Plot signal On; plt.gca() object

        '''

        ff, Pxx = signal.welch(signal, fs=self.fs)

        if ax is None:
            hold = False
            plt.figure(figsize=(15, 5))
            ax = plt.gca()
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('PSD [V**2/Hz]')
        else:
            hold = True

        ax.semilogy(ff, Pxx, label=title, **kwargs)
        plt.legend()

        if not hold:
            sns.despine()
            plt.show()
