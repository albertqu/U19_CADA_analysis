# U19_CADA_analysis

Python code repository for data analysis for U19 projects of Wilbrecht Lab. Analysis procedures include calcium signal processing, peristimulus visualization, GLM, neural decoding models and dimensionality models. 

**[CODE BASE UNDER CONSTRUCTION]**

## General Outline
The documentation consists of the following subparts:
* data file structure
* code base structure
* event type naming system (ETNS)
* example code
...

## Data File Structure
The file structure listed here suggests one possible analysis file configurations in local/server file structures and is open to modifications.

### General Setup
<pre>
- <root>
    - <b>CADA_data</b>: root for storing data of different modalities
        - <b>ProbSwitch_Raw</b>: Raw behavior data mats and other recording sources including photometry and binaries 
        - <b>ProbSwitch_FP_data</b>: ProbSwitch_FP_data: preprocessed data with behavior and FP dff ready for further analysis
            -<animal>_<session>: e.g. A2A-15B-B_RT_p151_session1_FP_RH
                - .mat: matlab consistent file saved with -v7.3 flag for hdf5 consistency
                - .hdf5: hdf5 file storing  
        - <b>RestaurantRow_Raw</b>: Restaurant row data
        - <b>RestaurantRow_FP_data</b>: preprocessed restaurant row
        ...
    - <b>CADA_plots</b>: root for storing plots of different sub-projects
        - <b>FP_NAc_D1D2_CADA</b>: root for plots for NAc
        - <b>FP_DMS_D1D2_CADA</b>: root for plots for DMS
        ...
</pre>

### ProbSwitch_FP_data
Data stored in this folders are organized by animal  and session names in individual folders:
```
 animal: A2A-15B-B_RT, session: p151_session1_FP_RH 
```
To load a filename in python code base one could use 
`encode_to_filename(folder, animal, session, ftypes)` function in `utils_loading.py` to obtain a dictionary (or string for single option) consisting of different file types. For instance to get the processed behavior mat of the session mentioned above we could use
```
folder = '<data_root>' # e.g. "<root>/CADA_data/ProbSwitch_FP_data/"
encode_to_filename(folder, 'A2A-15B-B_RT', 'p151_session1_FP_RH', 'behavior') # for more usage check specific code functions
```

### File Name Schemes
* **exper** .mat
* **behavior** .mat (processed from exper and synced with FP times)
* **bin_mat** binary file
* **green** green fluorescence
* **red** red FP
* **behavior** .mat behavior file
* **FP** processed dff hdf5 file
Check `decode_from_filename` in `utils.py` for more details for specific file name rule

## Code Base Structure
* `behaviors.py`:
    * Key input: processed behavior .mat files in `ProbSwitch_FP_data` folder
    * Key output: trial-based behavior features or behavior times and other relevant statistics like movement times
    * Key functions: (check specific code files for detailed descriptions)
        * `get_trial_features(mat, feature, as_array=False)`
        * `get_behavior_times(mat, behavior)`
    * **Special Note: the feature or behavior indexing follows the ETNS specified in the later section**
* `FP_deconv_test.py`
Tests regarding deconvolution algorithms on FP signals
* `modeling.py`
GLM/decoding models for FP data in ProbSwitch Tasks -- To be specified more
* `peristimulus.py`
    * Key input: dff traces and behavior times/features
    * Key output: peristimulus plots of various kinds indexed by different behavior events
    * Key functions: (check specific code files for detailed descriptions)
        * `align_activities_with_event(sigs, times, event_times, time_window, discrete=True, align_last=False)`
        * `behavior_aligned_FP_plots(folder, plots, behaviors, choices, options, zscore=True, base_method='robust', denoise=True)`
* `pipeline_FP_ProbSwitch.py`:
Consists of different short hand pipelines to run for different analysis using helper functions from other code files
* `script.py` *Please ignore, unorganized analysis ideas*
* `tests.py`
Various tests for different analysis steps, can be referred them for example function usage
* `utils.py`
Different utility functions for analysis, including, loading, preprocessing, simulation, filtering, visualizattion, process management
* `utils_models.py`
Utility functions for dimensionality reduction models and classifier/regression models


## Event Type Naming System (ETNS)
More specific details in documenations in `behaviors.py`
### Behavior Times
* `center_in`
* `center_out`
* `side_in`
* `outcome`
* `side_out`

### Event features:
* feature types:
    * `R`: Reward contingency of a trial, `'Rewarded', 'Unrewarded'`
    * `O`: Outcome contingency of a trial, `'Incorrect', 'Correct Omission', 'Rewarded'`
    * `A`: Action Laterality of the specific action (e.g. side out) **Note: this option will be modified for more careful action type grouping**
    * `S`: Suggests whether a trial occurs Periswitch, use `S[i]` to specify Preswitch with `0,1..i` steps, `S[-i]` to specify post switch with `0,-1..-i` steps 
    * `ITI`: inter trial interval bins

### bracket lag notation
* for behavior times; use `{t+i}` to select event times of `i` trials forward (`i>0`) or backward (`i<0`)
* for event features: it is perfectly valid to use chained notations to special a certain trial history: e.g. `R{t-3,t-2,t-1}` for 3 trial back reward history

## Example Code
The following code plot outcome times 
```python=
plot_type = 'trial_average'
folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
plots = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA"
zscore = True # Should not matter with 1 session
base_method = 'robust_fast'
denoise = True  # flag for whether to use wiener filter to clean noise from the source

for sg in ['Ca', 'DA']:
    choices = get_probswitch_session_by_condition(folder, group='all', region='NAc', signal=sg)
    # Plotting Option to generate a 1x2 plot for each session with column representing ipsi/contra port
    # and different color representing different trial outcomes, 
    # for DA/Ca separately (specified by `sg` in the outer loop)
    sigs = [sg]
    row = "FP"
    col = "A"
    hue = "O"
    rows = (sg, )
    cols = ('ipsi', 'contra')
    hues = ('Incorrect', 'Correct Omission', 'Rewarded')
    ylims = [[(-1.5, 2.1)] * 2] # specifies the ylimit showing on the subplots

    options = {'sigs': sigs, 'row': row, 'rows': rows, 'ylim': ylims,
               'col': col, 'cols': cols, 'hue': hue, 'hues': hues, 'plot_type': plot_type}
    behavior_aligned_FP_plots(folder, plots, 'outcome', choices, options,
                              zscore, base_method, denoise)
```


