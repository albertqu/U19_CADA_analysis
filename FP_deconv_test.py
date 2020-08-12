# System

# Data
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt

# Caiman
try:
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf.utilities import detrend_df_f
    from caiman.components_evaluation import estimate_components_quality_auto
    from caiman.source_extraction.cnmf import deconvolution
except ModuleNotFoundError:
    print("CaImAn not installed or environment not activated, certain functions might not be usable")
# Utils
from utils import *


def get_sample_spikes_slow_ramp():
    sp1 = np.full(1000, 0.02)
    sp1[50] = 0.2
    sp1[80] = 0.15
    sp1[60] = 0.15
    sp1[40] = 0.15
    sp1[20] = 0.15
    sp1[50:100] += np.arange(50) * 0.05 / 100
    sp1[150:200] += np.arange(50) * 0.05 / 100
    return sp1


def deconv_test1(sp1=None, impulse=True):
    g = np.array([ 0.87797883, -0.10934919])
    if impulse:
        sp1 = impulse_model_test(sp1, show=False)
    else:
        # simulated slow ramp test
        sp1 = get_sample_spikes_slow_ramp() if sp1 is None else sp1
    spc = SpikeCalciumizer(fmodel="AR_2", fluorescence_saturation=0, std_noise=0.02, alpha=1, g=g)
    rec = spc.binned_spikes_to_calcium(sp1.reshape((1, -1)))

    c2, bl2, c12, g2, sn2, sp2, lam2 = deconvolution.constrained_foopsi(rec.ravel(), p=2, bl=0,
                                                                        bas_nonneg=False, s_min=0)
    c2, bl2, c12, g2, sn2, sp3, lam2 = deconvolution.constrained_foopsi(rec.ravel(), p=2, bas_nonneg=False,
                                                                        s_min=0)

    conv = np.array([1] + list(-g2))
    h0 = inverse_kernel(conv, N=rec.shape[1], fft=True)
    x_hat, hhat = wiener_deconvolution(rec.ravel(), h0)

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True)
    axes[0].plot(sp1)
    axes[1].plot(sp3)
    axes[2].plot(sp2)
    axes[3].plot(x_hat)
    axes[4].plot(rec.ravel())
    axes[0].set_ylabel("truth spike")
    axes[1].set_ylabel("bl_auto")
    axes[2].set_ylabel('bl=0')
    axes[3].set_ylabel("wiener")
    axes[4].set_ylabel("calcium")

    # TODO: try band filtering


def impulse_model_test(sp1=None, show=True):
    g = np.array([0.87797883, -0.10934919])
    sp1 = get_sample_spikes_slow_ramp() if sp1 is None else sp1
    v1 = np.diff(sp1)
    a1 = np.diff(v1)
    a1_in = np.concatenate([[0, 0], a1])
    # v1 = np.gradient(sp1)
    # a1 = np.gradient(v1)
    spc = SpikeCalciumizer(fmodel='AR_2', g=[2, -1], std_noise=0.)
    spp = spc.binned_spikes_to_calcium(a1_in.reshape((1, -1))) + sp1[0]

    spc2 = SpikeCalciumizer(fmodel="AR_2", std_noise=0.02, g=g)
    c = spc2.binned_spikes_to_calcium(spp).ravel()

    # visualization
    if show:
        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True)
        axes[0].plot(sp1)
        axes[0].set_ylabel('true spike')
        axes[1].plot(v1)
        axes[1].set_ylabel("ds/dt")
        axes[2].plot(a1)
        axes[2].set_ylabel("d2s/dt2")
        axes[3].plot(spp.ravel())
        axes[3].set_ylabel("gradient-sim spikes")
        axes[4].plot(c)
        axes[4].set_ylabel("simulated calcium")
    return spp.ravel()


def wiener_deconv_test(y, diff=False):
    c, bl, c1, g, sn, sp, lam = deconvolution.constrained_foopsi(y, p=2, bas_nonneg=False, s_min=0)
    conv = np.array([1] + list(-g))
    h0 = inverse_kernel(conv, N=len(y), fft=True)
    x_hat, hhat = wiener_deconvolution(y, h0)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    if diff:
        axes[0].plot(np.vstack((np.diff(y, prepend=0), sp, x_hat)).T)
        axes[0].legend(['df/dt', 'deconv', 'wiener'])
    else:
        axes[0].plot(np.vstack((sp, x_hat)).T)
        axes[0].legend(['deconv', 'wiener'])
    axes[1].plot(y)
    axes[1].set_ylabel('dff')
    axes[2].plot(h0)
    axes[2].set_ylabel("IRF")



