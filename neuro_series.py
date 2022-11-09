import os.path

import pandas as pd

from packages.photometry_functions import get_zdFF, get_zdFF_old, jove_fit_reference
from peristimulus import *
from os.path import join as oj
import numbers


class FPSeries:
    """
    Wrapper class for manipulating fiber photometry data.

    Allows for recordings containing multiple synchronous ROIs, with one control channel interleaved
    with several signal channels.

    .fp_flags: dict
        desc: Map that provides guideline to decode multiplexed FP data,
            i.e. how to map csv flags to channels for
        format: `{trigger_mode: {channel: flag}}`
        e.g.: {'BSC1': {'415nm':1, '470nm':6},
               'BSC3': {'415nm':1, '470nm':2, '560nm':4}}
    .all_channels: dict
        desc: Map that maps each channel to a list of ROI names used in raw .neural_dfs
        format: `{channel: [rois]}`
        e.g.: {'470nm': ['left_470nm', 'right_470nm'],
               '415nm': ['left_410nm', 'right_410nm']}
    .sig_channels: dict
        desc: similar to .all_channels, excluding `control` channel (in .params['control'])

    hand shake process between bmat and neuro_series:
        neuro_series.realign(bmat)
        bmat.adjust_tmax(neuro_series)
    """
    io_specs = {'410nm': 'green',
                '415nm': 'green',
                '470nm': 'green',
                '560nm': 'red'}
    params = {'fr': 10,
              'time_unit': 'ms',
              'control': '410nm',
              'ignore_channels': []}
    fp_flags = {}
    quality_metric = 'aucroc'

    def __init__(self, data_file, ts_file, trig_mode, animal='test', session='0', hazard=0, cache_folder=None):
        self.neural_dfs = {}
        self.neural_df = None
        self.all_channels = {}
        self.sig_channels = {}
        self.animal, self.session = animal, session
        self.cache_folder = oj(cache_folder, animal, session) if (cache_folder is not None) else None
        self.hazard = hazard
        # TODO: add method to label quality for all ROIs

    def estimate_fr(self, ts):
        # First convert all the time units to seconds
        if self.params['time_unit'] == 'ms':
            ts = ts / 1000

    def __str__(self):
        return f"FP({self.animal},{self.session})"

    def calculate_dff(self, method, zscore=True, melt=True, **kwargs):
        # for all of the channels, calculate df using the method specified
        # TODO: add visualization technique by plotting the approximated
        # baseline against signal channels
        # Currently use original time stamps
        # method designed for lossy_ctrl merge channels method
        """
        Requires:
            self.sig_channels: for each channel, outline rois,
                e.g. {'415nm': ['green_415nm', 'red_415nm'], '470nm': ['green_470nm'], '560nm': ['red_560nm']}
            self.neural_df: output from self.merge_channels, pd.DataFrame with columns ['time'] + rois

        Returns: ZdF: making it method dependent, and store in a class variable
        """
        cache_file = None
        if self.cache_folder is not None:
            if not melt:
                aarg = '_unmelted'
            else:
                aarg = ''
            cache_file = oj(self.cache_folder, f"{self.animal}_{self.session}_dff_{method}{aarg}.pq")
            if os.path.exists(cache_file):
                return pd.read_parquet(cache_file)
        dff_name_map = {'iso_jove_dZF': 'jove'}
        iso_time = self.neural_df['time']
        dff_dfs = {'time': iso_time}
        for ch in self.sig_channels:
            for roi in self.sig_channels[ch]:
                if (self.hazard != 0) and (roi in self.hazard):
                    continue
                rec_time = self.neural_df['time'].values
                rec_sig = self.neural_df[roi].values
                iso_sig = self.neural_df[roi.replace(ch, self.params['control'])].values
                if method == 'dZF_jove':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF(iso_sig, rec_sig, remove=0)
                elif method == 'ZdF_jove':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF(iso_sig, rec_sig, remove=0)
                    dff = (dff - np.median(dff)) / np.std(dff)
                elif method == 'dZF_jove_raw':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF(iso_sig, rec_sig, remove=0, use_raw=True)
                elif method == 'dZF_jove_old':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF_old(iso_sig, rec_sig, remove=0)
                elif method == 'ZdF_jove_old':
                    assert zscore, 'isosbestic jove is always zscored'
                    dff = get_zdFF_old(iso_sig, rec_sig, remove=0, raw=True)
                    dff = (dff - np.mean(dff)) / np.std(dff)
                else:
                    dff = raw_fluor_to_dff(rec_time, rec_sig, iso_time, iso_sig,
                                           baseline_method=method, zscore=zscore, **kwargs)
                    prefix = 'ZdFF_' if zscore else 'dFF_'
                    method = prefix + method
                meas_name = 'ZdFF' if zscore else 'dFF'
                dff_dfs[roi +'_' +meas_name] = dff
                dff_dfs['method'] = [method] * len(dff)

        dff_df = pd.DataFrame(dff_dfs)
        if not melt:
            if cache_file is not None:
                if not os.path.exists(self.cache_folder):
                    os.makedirs(self.cache_folder)
                dff_df.to_parquet(cache_file)
            return dff_df
        id_labls = ['time', 'method']
        meas = np.setdiff1d(dff_df.columns, id_labls)
        melted = pd.melt(dff_df, id_vars=id_labls, value_vars=meas, var_name='roi', value_name=meas_name)
        melted['roi'] = melted['roi'].str.replace('_' +meas_name, '')
        if cache_file is not None:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
            melted.to_parquet(cache_file)
        return melted

    def merge_channels(self, ts_resamp_opt='lossy_ctrl'):
        # TODO: change default to interp
        # if unsatisfied with the direct merge we should interpolate
        # drop 200 frames, and then use ctrl timestamps
        # TODO: need a more rigorous check to make sure the jitter between times are within 50ms
        for ch in self.neural_dfs:
            if np.max(self.neural_dfs[ch].time) != self.neural_dfs[ch].time.values[-1]:
                logging.warning(f"max time not the same as the last frame time in {str(self)}, check data")
        drop_fr = 200
        if ts_resamp_opt == 'lossy_ctrl':
            min_len = min(len(self.neural_dfs[ch]) for ch in self.neural_dfs)
            ctrl_ch = self.params['control']
            time_axis = self.neural_dfs[ctrl_ch][drop_fr:min_len].reset_index(drop=True).time
            all_dfs = [time_axis]
            chtime_dfs = []
            for ch in self.neural_dfs:
                dcs = [c for c in self.neural_dfs[ch].columns if c != 'time']
                sub_df = self.neural_dfs[ch].rename(
                    columns={'time': 'time_' +ch})[drop_fr:min_len].reset_index(drop=True)
                all_dfs.append(sub_df[dcs])
                if ch != ctrl_ch:
                    ch_time = sub_df['time_' +ch]
                    if not np.all(np.abs(ch_time - time_axis) < 50):
                        print(f'{ch} has large time offset from control channels after preprocessing')
                    chtime_dfs.append(ch_time)
            self.neural_df = pd.concat(all_dfs +chtime_dfs, axis=1)
            if np.max(np.abs(self.neural_df['time'].diff())) > 75:
                logging.warning('warning! significant deviation from 50ms in final sampling interval')
        elif ts_resamp_opt == 'interp':
            neural_dfs = self.neural_dfs
            min_ch = min(neural_dfs.keys(), key=lambda k: np.max(neural_dfs[k].time))
            time_axis = neural_dfs[min_ch].time[drop_fr:]
            neural_df = {'time': time_axis.values}
            for src in neural_dfs:
                src_rois = [c for c in neural_dfs[src].columns if c != 'time']
                for src_roi in src_rois:
                    sig = neural_dfs[src][src_roi].values
                    sigtime = neural_dfs[src].time.values
                    assert np.all(np.diff(sigtime) > 0), 'signal reverse order'
                    if src == min_ch:
                        neural_df[src_roi] = sig[drop_fr:]
                    else:
                        neural_df[src_roi] = interpolate.interp1d(sigtime, sig, fill_value='extrapolate')(time_axis)
            # when dataframe is created from dfs indexes are kept
            neural_df = pd.DataFrame(neural_df)
            self.neural_df = neural_df
        else:
            raise RuntimeError(f'Unknown resampling method {ts_resamp_opt}')

    def realign_time(self, reference=None):
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)
        else:
            if reference is None:
                zero = min(self.neural_dfs[ch].time for ch in self.neural_dfs)
            else:
                zero = reference
            assert isinstance(zero, numbers.Number), \
                f'reference has to be BehaviorMat, number, or None but found {type(reference)}'
            transform_func = lambda ts: ts - zero
        for ch in self.neural_dfs:
            self.neural_dfs[ch]['time'] = transform_func(self.neural_dfs[ch]['time'])
        if self.neural_df is not None:
            for c in self.neural_df.columns:
                if 'time' in c:
                    self.neural_df[c] = transform_func(self.neural_df[c])
        # update time unit to match BehaviorMat

    def diagnose_multi_channels(self, viz=True, plot_path=None):
        # Step 1: Visualize the Discontinuity in first 5 min
        # whole session visualization
        # Step 2:
        time_axis = self.neural_df['time']
        control_ch = self.params['control']
        sig_scores = {}
        fig_tag = f'{self.animal} {self.session}'
        for ch in self.sig_channels:
            for roi in self.sig_channels[ch]:
                raw_reference = self.neural_df[roi.replace(ch, control_ch)].values
                raw_signal = self.neural_df[roi].values
                # do not smooth based on frame rate
                fig, sig_score, _ = FP_quality_visualization(raw_reference, raw_signal, time_axis, drop_frame=200,
                                                             time_unit=self.params['time_unit'],
                                                             sig_channel=ch, control_channel=control_ch,
                                                             roi=roi, tag=fig_tag, viz=viz)
                if viz:
                    fig2 = FP_viz_whole_session(raw_reference, raw_signal, time_axis, interval=600, drop_frame=200,
                                                time_unit=self.params['time_unit'], sig_channel=ch, control_channel=control_ch,
                                                roi=roi, tag=fig_tag)
                if fig is not None and plot_path is not None:
                    animal_folder = oj(plot_path, self.animal)
                    animal, session = self.animal, self.session
                    if not os.path.exists(animal_folder):
                        os.makedirs(animal_folder)
                    fig.savefig(oj(animal_folder,
                                   f'{animal}_{session}_{roi}_quality_{self.quality_metric}.png'))
                    fig2.savefig(oj(animal_folder, f'{animal}_{session}_{roi}_raw_whole_session.png'))
                    plt.close(fig)
                    plt.close(fig2)
                sig_scores[roi] = sig_score
        return sig_scores


class BonsaiFP3001(FPSeries):
    # Currently this just works for RR, extend to probswitch later by exploring raw data

    fp_flags = {'TRIG1': {'410nm': 1, '470nm': 6},
                'TRIG3': {'410nm': 1, '470nm': 2, '560nm': 4}}

    rois = []

    def __init__(self, data_file, ts_file, trig_mode, animal='test', session='0', hazard=0, cache_folder=None):
        # determine the golden standard for resampling time series
        super().__init__(data_file, ts_file, trig_mode, animal, session, hazard, cache_folder)
        data = pd.read_csv(data_file, skiprows=1, names=['frame', 'cam_time', 'flag'] + self.rois)
        data_ts = pd.read_csv(ts_file, names=['time'])
        data_fp = pd.concat([data, data_ts.time], axis=1)
        trig_flags = self.fp_flags[trig_mode]
        self.neural_dfs = {src: None for src in trig_flags}
        self.sig_channels = {}
        self.all_channels = {}
        for src, flg in trig_flags.items():
            if src not in self.params['ignore_channels']:
                channel = self.io_specs[src]
                src_rois = [roi for roi in self.rois if (channel in roi)]
                new_names = {roi: roi.replace(channel, src) for roi in src_rois}
                self.all_channels[src] = list(new_names.values())
                if src != self.params['control']:
                    self.sig_channels[src] = list(new_names.values())
                self.neural_dfs[src] = pd.DataFrame(data_fp.loc[data_fp.flag==flg,
                                                                ['time'] + src_rois]).rename(columns=new_names)


class OldFP3001(FPSeries):
    # Assumes unilateral
    fp_flags = {'BSC1': {'415nm' :1, '470nm' :6},
                'BSC3': {'415nm' :1, '470nm' :2, '560nm' :4}}
    rois = ['green']

    params = {'fr': 20,
              'time_unit': 's',
              'control': '415nm',
              'ignore_channels': []}

    def __init__(self, data_file, ts_file=None, trig_mode=None, animal='test', session='0', hazard=0):
        super().__init__(data_file, ts_file, trig_mode, animal, session)
        self.sig_channels = {}
        if trig_mode is None:
            trig_mode ='BSC1'
        if data_file.split('.')[1] == 'mat':
            with h5py.File(data_file, 'r') as neural_data:
                for src in self.fp_flags[trig_mode]:
                    self.neural_dfs[src] = pd.DataFrame(
                        {'time': np.array(neural_data[f'FP_out/time{src[:-2]}']).ravel(),
                          src: np.array(neural_data[f'FP_out/sig{src[:-2]}']).ravel()})
                    if src != self.params['control']:
                        self.sig_channels[src] = [src]


class BonsaiRR2Hemi2Ch(BonsaiFP3001):
    rois = ['right_red', 'left_red', 'right_green', 'left_green']

    params = {'fr': 20,
              'time_unit': 's',
              'control': '410nm',
              'ignore_channels': []}

    def __init__(self, data_file, ts_file, trig_mode, animal='test', session='0', hazard=0, cache_folder=None):
        super().__init__(data_file, ts_file, trig_mode, animal, session, hazard, cache_folder)
        if hazard == -1:
            self.hazard = ['left_470nm']
        elif hazard == -2:
            self.hazard = ['right_470nm']


class BonsaiPS1Hemi2Ch(BonsaiFP3001):

    fp_flags = {'BSC1': {'415nm' :1, '470nm' :6},
                'BSC3': {'415nm' :1, '470nm' :2, '560nm' :4}}

    rois = ['green'] # better represent

    params = {'fr': 20,
              'time_unit': 's',
              'control': '415nm',
              'ignore_channels': []}

    def __init__(self, data_file, ts_file, trig_mode, animal='test', session='0', hazard=0, cache_folder=None):
        if pd.isnull(trig_mode):
            trig_mode = 'BSC1'
        super().__init__(data_file, ts_file, trig_mode, animal, session, hazard, cache_folder)
        if hazard == -1:
            self.hazard = list(np.concatenate([[src_roi for src_roi in self.sig_channels[src]] for src in self.sig_channels]))


