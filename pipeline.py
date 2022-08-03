import os

from modeling import *
from peristimulus import *
import shutil
import seaborn as sns
from scipy.stats import spearmanr, shapiro
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib as mpl

plt.subplots_adjust(wspace=0.2, hspace=0.3)
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams["legend.fontsize"] = "x-large"
mpl.rcParams['figure.titlesize'] = 22


def pipeline_example():
    experiment = 'ProbSwitch'
    root = "/content/drive/MyDrive/WilbrechtLab/U19_project/analysis/"
    folder = get_file_path_by_experiment(experiment, root)
    print(get_probswitch_session_by_condition(folder, group='all', region='NAc', signal='DA'))
    folder = "/Volumes/ALBERTSHD/WilbrechtLab/CADA_data/ProbSwitch_FP_data"
    # folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plot_out = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA/belief_state"
    time_window_dict = {'center_in': np.arange(-500, 501, 50),
                        'center_out': np.arange(-500, 501, 50),
                        'outcome': np.arange(-500, 2001, 50),
                        'side_out': np.arange(-500, 1001, 50)}

    animal, session = 'A2A-16B-1_RT', 'p147_FP_LH'
    region = 'NAc'
    event_FP_mat = animal_session_to_event_mat(folder, animal, session, 'NAc', prepost=(-4, 4), event='outcome')
    modeling_pdf = get_modeling_pdf(folder, animal, session)
    giant_mat = pd.concat([event_FP_mat, modeling_pdf], axis=1)


def reg_filestruct_BSDML_FP(folder):
    for animal in os.listdir(folder):
        animal_path = os.path.join(folder, animal)
        if os.path.isdir(animal_path):
            animal_parts = animal.split('-')
            if (len(animal_parts) == 3) and ('_' not in animal_parts[-1]):
                animal_ID = f'{animal_parts[0]}-{animal_parts[1]}_{animal_parts[2]}'
                new_animal_path = os.path.join(folder, animal_ID)
                os.rename(animal_path, new_animal_path)
                for f in os.listdir(new_animal_path):
                    if animal in f:
                        os.rename(os.path.join(new_animal_path, f), os.path.join(new_animal_path,
                                                                                 f.replace(animal,
                                                                                           animal_ID)))


def get_file_path_by_experiment(expr, root):
    if expr == 'ProbSwitch_Chris':
        folder = os.path.join(root, 'ProbSwitch/ProbSwitch_FP_data')
    elif expr == 'BSDML':
        folder = os.path.join(root, 'ProbSwitch/ProbSwitch_FP_data')
    elif expr == 'restaurant_row':
        folder = os.path.join(root, 'rr_data/ProbSwitch_FP_data')


def pseudo_pipeline():
    experiment = 'ProbSwitch'
    root = "/content/drive/MyDrive/WilbrechtLab/U19_project/analysis/"
    folder = get_file_path_by_experiment(experiment, root)
    plot_out = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA/belief_state"
    time_window_dict = {'center_in': np.arange(-500, 501, 50),
                        'center_out': np.arange(-500, 501, 50),
                        'outcome': np.arange(-500, 2001, 50),
                        'side_out': np.arange(-500, 1001, 50)}

    data = pd.read_csv(fp_file, skiprows=1, names=[
                       'frame', 'cam_time_stamp', 'flag', 'right_red', 'left_red', 'right_green', 'left_green'])
    data_time_stamps = pd.read_csv(
        fp_time_stamps, names=['time_stamps'])

    data_fp = pd.concat([data, data_time_stamps.time_stamps], axis=1)

    fp_flag = {'BSC1': {'control':1, 'green':6},
               'BSC3': {'control':1, 'green':2, 'red':4}}
    fp_flag = fp_flag[trigger_mode]

    animal, session = 'A2A-16B-1_RT', 'p147_FP_LH'
    asession_data = get_animal_session_data()
    bmat = BehaviorMat(animal, session)
    FP_df = load_fp_df()  # refer to Laura code
    FP_df = resync_timestamp(FP_df, ts_from, ts_to)


def get_animal_session_data():
    pass


def reorganize_BSD_filenames():
    # root = r"Z:\2ABT\ProbSwitch\BSDML_exper"
    root = r"Z:\2ABT\ProbSwitch\BSDML_FP\barrier"
    namemap = {"D1-R34_RT": "BSD011",
               "D1-R34_LT":	"BSD012",
               "D1-R35_RV":	"BSD013"}

    # namemap = {'A2A-15B_RT': 'BSD002',
    #            'A2A-15B-B_RT': 'BSD002',
    #            'A2A-16B-1_RT': 'BSD003',
    #            'A2A-16B_RT': 'BSD003',
    #            'A2A-16B-1_TT': 'BSD004',
    #            'A2A-16B_TT': 'BSD004',
    #            'D1-27H_LT':	'BSD005'
    #            }
    namemap = {'A2A-15B_RT': 'BSD002',
               'A2A-15B-B_RT': 'BSD002',
               'D1-27H_LT':	'BSD005'
               }
    from utils_system import rename_dir_files_recursive
    rename_dir_files_recursive(root, namemap)


def reorganize_BSD_Chris_filenames():
    # root = r"Z:\2ABT\ProbSwitch\BSDML_exper"
    root = r"Z:\Alumni\Chris Hall\Belief State"
    outpath = r'Z:\2ABT\ProbSwitch\Chris_Raw'
    folders = [os.path.join(root, 'BeliefState_FP_FCcomp'),
               os.path.join(root, 'FP_BeliefState_SMAcomp'),
               os.path.join(root, 'BeliefState_ProbSwitch')]

    # namemap = {'A2A-15B_RT': 'BSD002',
    #            'A2A-15B-B_RT': 'BSD002',
    #            'A2A-16B-1_RT': 'BSD003',
    #            'A2A-16B_RT': 'BSD003',
    #            'A2A-16B-1_TT': 'BSD004',
    #            'A2A-16B_TT': 'BSD004',
    #            'D1-27H_LT':	'BSD005'
    #            }
    namemap = {'A2A-15B_RT': 'BSD002',
               'A2A-15B-B_RT': 'BSD002',
               'D1-27H_LT':	'BSD005'
               }
    for animal in namemap:
        alias = namemap[animal]
        animal_folder = os.path.join(outpath, alias)
        if not os.path.exists(animal_folder):
            os.makedirs(animal_folder)
        for folder in folders:
            for f in os.listdir(folder):
                if (animal in f):
                    new_fname = f.replace(animal, alias)
                    options = decode_from_filename(f)
                    if options is None:
                        print('Error with', f)
                    print(options['animal'], options['session'], options['H'])
                    # shutil.copy2(os.path.join(folder, f), os.path.join(animal_folder, new_fname))


class BSDML_Expr:
    # EXPERIMENT specific loading
    file_template = {}
