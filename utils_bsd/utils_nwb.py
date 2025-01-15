import pandas as pd
import re, os
from datetime import datetime
import pytz
import scipy.io as sio
import logging
import platform
import uuid

import pynwb
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from datetime import datetime
from os.path import join as oj
from ndx_fiber_photometry import (
    Indicator,
    OpticalFiber,
    ExcitationSource,
    Photodetector,
    FiberPhotometryTable,
    FiberPhotometry,
    FiberPhotometryResponseSeries,
    DichroicMirror,
    BandOpticalFilter,
)
from pynwb.ophys import RoiResponseSeries

####################################################
############### General NWB functions ##############
####################################################

def pse_get_session_start(pse, animal, session, raw_folder):
    """ 3 steps:
    1. first look in meta to see if it has start time. 
    2. then look at FP time
    3. then look at exper
    """

    tz = pytz.timezone('America/Los_Angeles')
    session_dict = pse.meta[(pse.meta['animal'] == animal) & 
                (pse.meta['session'] == session)].iloc[0].to_dict()
    try:
        session_start_date = session_dict.get('date')
        session_start_time = session_dict.get('time_in')
        session_start = session_start_date + '-' + session_start_time
        return datetime.strptime(session_start, 
                                '%m/%d/%Y-%H:%M').replace(tzinfo=tz)
    except:
        fp_name = pse.encode_to_filename(animal, session, 'FP_')
        dt = fpname_get_datetime(fp_name)
        if dt:
            return dt
        else:
            animal, animal_ID = session_dict['animal'], session_dict['animal_ID']
            f_match1 = [f for f in os.listdir(raw_folder) if animal in f and session in f]
            f_match2 = [f for f in os.listdir(raw_folder) if animal_ID in f and session in f]
            f_match = f_match1 + f_match2
            if f_match:
                experf = oj(raw_folder, f_match[0])
                # print(os.path.getctime(experf))
            else:
                return
            try:
                mat_data = sio.loadmat(experf)
                t_array = mat_data['exper']['control'][0, 0]['param'][0, 0]['start'][0, 0]['value'][0, 0][0]
                first_five = list(t_array[:5].astype(int))
                dt = datetime(*first_five, int(t_array[5]), tzinfo=tz)
            except:
                dt = datetime.fromtimestamp(file_creation_time(experf), tz=tz)
    
def fpname_get_datetime(fp_name):
    pattern = r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}'
    # Extract the timestamp
    match = re.search(pattern, fp_name)
    if match:
        session_start = datetime.strptime(match.group(), 
                                        '%Y-%m-%dT%H_%M_%S')
        session_start = session_start.replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        return session_start
    else:
        return None
    

def file_creation_time(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            logging.warning("Creation date is not available, using modification date")
            return stat.st_mtime

def df_col_safe_access(df, col, i):
    if col in df.columns:
        return df[col].iat[i]
    else:
        return None

def add_trials_from_bdf(nwbfile, bdf, start_time_col='center_in', description_map=None):
    """
    Every nwb file has a trials table for storing trial info about
    the session. Each row in the table must have a start_time and a
    stop_time value encoding the trial interval. As these fields are not
    explicitly specified in the bdf, I took the center_in time to be the start
    of the trial and the largest time (i.e. float) in the row to be the end of the trial.
    Adaptation credit: Mohammed Osman
    """
    
    for col in bdf.columns:
        if col in description_map:
            desc = description_map[col]
            if bdf[col].dtype == object or bdf[col].dtype == 'category':
                bdf[col] = bdf[col].astype(str)
            nwbfile.add_trial_column(name=col, description=desc)    # TODO: maybe add a better description
    for i, row in bdf.iterrows():
        start_time = row[start_time_col]
        stop_time = max(t for t in row if isinstance(t, float))
        row_dict = row.to_dict()
        nwbfile.add_trial(start_time, stop_time, **{k: row_dict[k] for k in description_map})
    

def add_photometry_simple(nwbfile, dff, raw_415, raw_470):
    """
    For the timebeing, we can simply store the photometry data using a generic
    TimeSeries object. We will store the raw and processed data in the acquisition
    and processing fields of the nwb file, respectively.
    credit: Mohammed Osman
    """
    raw_415_ts = TimeSeries(
        name='raw_415', unit='F',
        data=raw_415['415nm'].to_numpy(),
        timestamps=raw_415['time'].to_numpy()
    )
    raw_470_ts = TimeSeries(
        name='raw_470', unit='F',
        data=raw_470['470nm'].to_numpy(),
        timestamps=raw_470['time'].to_numpy()
    )
    dff_ts = TimeSeries(
        name='dff', unit='ZdFF',
        data=dff['ZdFF'].to_numpy(),
        timestamps=dff['time'].to_numpy()
    )
    
    nwbfile.add_acquisition(raw_415_ts)
    nwbfile.add_acquisition(raw_470_ts)
    ophys_module = nwbfile.create_processing_module(
        name='ophys', description=f'processed fiber photometry data ({dff["method"][0]})'
    )
    ophys_module.add(dff_ts)


###############################################
########### BSD project specific ##############
###############################################

def nwbfile_to_bdf(nwbfile):
    """
    Given the nwbfile object, this function will convert the nwb trial table
    back into the behavioral data frame (bdf). As the trial table is essentially
    just the bdf with start_time and stop_time columns, we just cast the trial
    table as a dataframe and remove said columns. If the presence of the start_time
    and stop_time columns isn't an issue, the whole table can be retrieved by
    calling nwbfile.trials.to_dataframe()
    """
    bdf = nwbfile.trials.to_dataframe()
    bdf.drop('start_time', axis=1, inplace=True)
    bdf.drop('stop_time', axis=1, inplace=True)
    return bdf

def bsd_description_map():
    time_var = 'time (relative to session start)'
    description_map = {
        'animal': 'mouse in task',
        'session': 'session names as pXX for postnatal date',
        'trial': 'trial number',
        'center_in': f'{time_var} when mouse pokes in center port to start a new trial',
        'center_out': f'{time_var} when mouse leaves the center port',
        'side_in': f'{time_var} when mouse pokes in side port',
        'outcome': f'{time_var} of the outcome',
        'zeroth_side_out': f'{time_var} when mouse triggers first IR beam after outcome',
        'first_side_out': f'{time_var} when mouse leaves the side port for the first time after outcome',
        'last_side_out': f'{time_var} right before mouse pokes in center port to start a new trial',
        'action': 'left or right',
        'rewarded': 'rewarded or not',
        'trial_in_block': 'trial number in block',
        'prebswitch_num': 'number of trials prior to block switch',
        'block_num': 'block number',
        'state': 'true block state',
        'last_side_out_side': 'left or right',
        'tmax': f'max {time_var} in session'
    }
    return description_map


def add_photometry_bsd(nwbfile, ps_series, hemi, location):
    coord = (1.2, 1.2, 4.1)
    # adapted from https://github.com/catalystneuro/ndx-fiber-photometry/tree/main
    fiber_photometry_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="fiber photometry table",
    )
    indicator = Indicator(
        name=f"indicator",
        description="Calcium indicator at 470nm and isosbestic point at 415nm",
        label="GCamp8f",
        injection_location=f'{hemi}_{location}',
        injection_coordinates_in_mm=coord,
    )
    optical_fiber = OpticalFiber(
        name="optical_fiber",
        model="fiber_model",
        numerical_aperture=0.37,
        core_diameter_in_um=400.0,
    )

    registers = [indicator, optical_fiber]

    for i, channel in enumerate(['415nm', '470nm']):
        wavelen = float(channel.replace('nm', ''))
        excitation_src = ExcitationSource(
            name=f"excitation_source_{i}", 
            description=f"excitation sources for {channel}",
            model="laser model",
            illumination_type="laser",
            excitation_wavelength_in_nm=wavelen
        )
        photodector = Photodetector(
            name=f"photodetector_{i}",
            description="CMOS detector",
            detector_type="CMOS",
            detected_wavelength_in_nm=wavelen,
            gain=100.0,
        )

        dichroic_mirror = DichroicMirror(
            name=f"dichroic_mirror_{i}",
            description="Dichroic mirror for red indicator",
            model="dicdichroic mirror model",
            cut_on_wavelength_in_nm=-1.0,
            transmission_band_in_nm=(-1.0, -1.0),
            cut_off_wavelength_in_nm=-1.0,
            reflection_band_in_nm=(-1.0, -1.0),
            angle_of_incidence_in_degrees=45.0,
        )

        band_optical_filter = BandOpticalFilter(
            name=f"band_optical_filter_{i}",
            description="check Neurophotometry docs",
            model="emission filter model",
            center_wavelength_in_nm=-1.0,
            bandwidth_in_nm=-1.0,
            filter_type="Bandpass",
        )

        registers = registers + [excitation_src, photodector, dichroic_mirror, band_optical_filter]

        fiber_photometry_table.add_row(
            location=f"{hemi}_{location}",
            coordinates=(1.2, 1.2, 4.1), # Indicator injection location in stereotactic coordinates (AP, ML, DV) mm relative to Bregma.
            indicator=indicator,
            optical_fiber=optical_fiber,
            excitation_source=excitation_src,
            photodetector=photodector,
            dichroic_mirror=dichroic_mirror,
            emission_filter=band_optical_filter,
        )

    for i, channel in enumerate(['415nm', '470nm']):
        channel_df = ps_series.neural_dfs[channel]

        fp_roi_series = FiberPhotometryResponseSeries(
            name=f"fp_series_{channel}",
            description=f"{channel} series",
            data=channel_df[channel].values,
            timestamps=channel_df['time'].values,
            unit="F",
            fiber_photometry_table_region=fiber_photometry_table.create_fiber_photometry_table_region(
                region=[i], description=f'{channel}')
        )
        nwbfile.add_acquisition(fp_roi_series)
        # series_roi = RoiResponseSeries(
        #     name = f'raw_{channel}',
        #     data = channel_df[channel].values,
        #     timestamps = channel_df['time'].values, # relative to session start in seconds
        #     unit='F',
        #     rois = fiber_photometry_table.create_fiber_photometry_table_region(
        #         region=[i], description=f'{channel}')
        # )
        # nwbfile.add_acquisition(series_roi)

    for reg in registers:
        nwbfile.add_device(reg)

    # skip dff for now
    # ophys_module = nwbfile.create_processing_module(
    #     name='ophys', description=f'processed fiber photometry data ({dff["method"][0]})'
    # )
    # ophys_module.add(dff_ts)

    nwbfile.add_lab_meta_data(
        FiberPhotometry(
        name="fiber_photometry",
        fiber_photometry_table=fiber_photometry_table)
    )

def convert_to_nwb_BSD(pse, animal, session, raw, out):
    """ 
    Inspired by NWB documentation and Mohammed Osman's code

    Bug: after saving nwb file automatically pynwb automatically adds irrelevant fields like electrode_groups
    """
    # session loading
    bmat, ps_series = pse.load_animal_session(animal, session)
    session_start = pse_get_session_start(pse, animal, session, raw)
    if pd.isnull(session_start):
        logging.warning(f"Could not find session start time for {animal}, {session}, this mean no exper file!")

    session_dict = pse.meta[(pse.meta['animal'] == animal) & 
                (pse.meta['session'] == session)].iloc[0].to_dict()
    
    nwbfile = NWBFile(
        session_description='Probswitch experiment for 2ABT',                 
        identifier=str(uuid.uuid4()),                 
        session_start_time=session_start, 
        session_id = session,
        lab= 'Wilbrecht Lab',
        institution='UC Berkeley',
        keywords=['2ABT', 'bayesian', 'dopamine'],
        related_publications=['https://www.biorxiv.org/content/10.1101/2023.11.10.566306v2']
    )

    subj = session_dict.get('animal')
    weight = session_dict.get('pre_task_weight')
    nwbfile.subject = pynwb.file.Subject(
        age='P{}D'.format(int(session_dict.get('age'))),    # difference between curr_age and age?
        description='mouse subject with dLight fluorophore in NAc',
        sex=session_dict.get('sex', 'NA'),
        species='Mus musculus',
        subject_id=subj,
        weight=f'{weight} g' if weight else 'NA',
        # post_task_weight=session_dict.get('post_task_weight'),
        # strain=None
    )
    device = nwbfile.create_device(
        name="dual-color fiber photometry",
        description="Neurophotonics fiber photometry system",
        manufacturer="Neurophotonics",
    )
    hemi = session_dict.get('hemi')
    location = 'NAc'
    desc_map = bsd_description_map()
    add_trials_from_bdf(nwbfile, bmat.todf(), start_time_col='center_in', description_map=desc_map)
    if ps_series is not None:
        add_photometry_bsd(nwbfile, ps_series, hemi, location)
    
    outfolder = oj(out, animal)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outname = oj(outfolder, f'{animal}_{session}.nwb')
    with NWBHDF5IO(outname, 'w') as io:
        io.write(nwbfile)
    return outname, nwbfile
