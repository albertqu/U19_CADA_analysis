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
import sys
from datetime import datetime

plt.subplots_adjust(wspace=0.2, hspace=0.3)
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = "x-large"
mpl.rcParams["figure.titlesize"] = 22


def pipeline_example():
    experiment = "ProbSwitch"
    root = "/content/drive/MyDrive/WilbrechtLab/U19_project/analysis/"
    folder = get_file_path_by_experiment(experiment, root)
    print(
        get_probswitch_session_by_condition(
            folder, group="all", region="NAc", signal="DA"
        )
    )
    folder = "/Volumes/ALBERTSHD/WilbrechtLab/CADA_data/ProbSwitch_FP_data"
    # folder = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_data"
    plot_out = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA/belief_state"
    time_window_dict = {
        "center_in": np.arange(-500, 501, 50),
        "center_out": np.arange(-500, 501, 50),
        "outcome": np.arange(-500, 2001, 50),
        "side_out": np.arange(-500, 1001, 50),
    }

    animal, session = "A2A-16B-1_RT", "p147_FP_LH"
    region = "NAc"
    event_FP_mat = animal_session_to_event_mat(
        folder, animal, session, "NAc", prepost=(-4, 4), event="outcome"
    )
    modeling_pdf = get_modeling_pdf(folder, animal, session)
    giant_mat = pd.concat([event_FP_mat, modeling_pdf], axis=1)


def reg_filestruct_BSDML_FP(folder):
    for animal in os.listdir(folder):
        animal_path = os.path.join(folder, animal)
        if os.path.isdir(animal_path):
            animal_parts = animal.split("-")
            if (len(animal_parts) == 3) and ("_" not in animal_parts[-1]):
                animal_ID = f"{animal_parts[0]}-{animal_parts[1]}_{animal_parts[2]}"
                new_animal_path = os.path.join(folder, animal_ID)
                os.rename(animal_path, new_animal_path)
                for f in os.listdir(new_animal_path):
                    if animal in f:
                        os.rename(
                            os.path.join(new_animal_path, f),
                            os.path.join(new_animal_path, f.replace(animal, animal_ID)),
                        )


def get_file_path_by_experiment(expr, root):
    if expr == "ProbSwitch_Chris":
        folder = os.path.join(root, "ProbSwitch/ProbSwitch_FP_data")
    elif expr == "BSDML":
        folder = os.path.join(root, "ProbSwitch/ProbSwitch_FP_data")
    elif expr == "restaurant_row":
        folder = os.path.join(root, "rr_data/ProbSwitch_FP_data")


def pseudo_pipeline():
    experiment = "ProbSwitch"
    root = "/content/drive/MyDrive/WilbrechtLab/U19_project/analysis/"
    folder = get_file_path_by_experiment(experiment, root)
    plot_out = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_plots/FP_NAc_D1D2_CADA/belief_state"
    time_window_dict = {
        "center_in": np.arange(-500, 501, 50),
        "center_out": np.arange(-500, 501, 50),
        "outcome": np.arange(-500, 2001, 50),
        "side_out": np.arange(-500, 1001, 50),
    }

    data = pd.read_csv(
        fp_file,
        skiprows=1,
        names=[
            "frame",
            "cam_time_stamp",
            "flag",
            "right_red",
            "left_red",
            "right_green",
            "left_green",
        ],
    )
    data_time_stamps = pd.read_csv(fp_time_stamps, names=["time_stamps"])

    data_fp = pd.concat([data, data_time_stamps.time_stamps], axis=1)

    fp_flag = {
        "BSC1": {"control": 1, "green": 6},
        "BSC3": {"control": 1, "green": 2, "red": 4},
    }
    fp_flag = fp_flag[trigger_mode]

    animal, session = "A2A-16B-1_RT", "p147_FP_LH"
    asession_data = get_animal_session_data()
    bmat = BehaviorMat(animal, session)
    FP_df = load_fp_df()  # refer to Laura code
    FP_df = resync_timestamp(FP_df, ts_from, ts_to)


def get_animal_session_data():
    pass


"""####################################################
################### File Structure ####################
####################################################"""


# Probswitch experiments
def reorganize_BSD_filenames():
    # root = r"Z:\2ABT\ProbSwitch\BSDML_exper"
    root = r"Z:\2ABT\ProbSwitch\BSDML_FP\barrier"
    namemap = {"D1-R34_RT": "BSD011", "D1-R34_LT": "BSD012", "D1-R35_RV": "BSD013"}

    # namemap = {'A2A-15B_RT': 'BSD002',
    #            'A2A-15B-B_RT': 'BSD002',
    #            'A2A-16B-1_RT': 'BSD003',
    #            'A2A-16B_RT': 'BSD003',
    #            'A2A-16B-1_TT': 'BSD004',
    #            'A2A-16B_TT': 'BSD004',
    #            'D1-27H_LT':	'BSD005'
    #            }
    namemap = {"A2A-15B_RT": "BSD002", "A2A-15B-B_RT": "BSD002", "D1-27H_LT": "BSD005"}
    from utils_system import rename_dir_files_recursive

    rename_dir_files_recursive(root, namemap)


def reorganize_BSD_Chris_filenames():
    # root = r"Z:\2ABT\ProbSwitch\BSDML_exper"
    root = r"Z:\Alumni\Chris Hall\Belief State"
    outpath = r"Z:\2ABT\ProbSwitch\Chris_Raw"
    folders = [
        os.path.join(root, "BeliefState_FP_FCcomp"),
        os.path.join(root, "FP_BeliefState_SMAcomp"),
        os.path.join(root, "BeliefState_ProbSwitch"),
    ]

    # namemap = {'A2A-15B_RT': 'BSD002',
    #            'A2A-15B-B_RT': 'BSD002',
    #            'A2A-16B-1_RT': 'BSD003',
    #            'A2A-16B_RT': 'BSD003',
    #            'A2A-16B-1_TT': 'BSD004',
    #            'A2A-16B_TT': 'BSD004',
    #            'D1-27H_LT':	'BSD005'
    #            }
    namemap = {
        "A2A-15B_RT": "BSD002",
        "A2A-15B-B_RT": "BSD002",
        "A2A-16B-1_RT": "BSD003",
        "A2A-16B_RT": "BSD003",
        "A2A-16B-1_TT": "BSD004",
        "A2A-16B_TT": "BSD004",
        "D1-27H_LT": "BSD005",
        "A2A-19B_LT": "BSD006",
        "A2A-19B_RT": "BSD007",
        "A2A-19B_RV": "BSD008",
        "D1-28B_LT": "BSD009",
    }

    for animal in namemap:
        alias = namemap[animal]
        animal_folder = os.path.join(outpath, alias)
        if not os.path.exists(animal_folder):
            os.makedirs(animal_folder)
        for folder in folders:
            for f in os.listdir(folder):
                if animal in f:
                    new_fname = f.replace(animal, alias)
                    target_file = os.path.join(animal_folder, new_fname)
                    if not os.path.exists(target_file):
                        options = decode_from_filename(f)
                        if options is None:
                            print("Error with", f)
                        # else:
                        #     print(options['animal'], options['session'], options['H'])
                        shutil.copy2(
                            os.path.join(folder, f),
                            os.path.join(animal_folder, new_fname),
                        )


## Restaurant rows
def organize_RR_structures(root, out=None, fp=False):
    from os.path import join as oj

    name_map = {
        "behavior": r"^RR_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "FP": r"^FP_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "FPTS": r"^FPTS_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
    }
    animals = [
        # "RRM042",
        # "RRM051",
        # "RRM052",
        "RRM054",
        "RRM056",
        # "RRM048",
        # "RRM049",
    ]  # 'RRM042', 'RRM043', 'RRM044', 'RRM045'
    if out is None:
        out = oj(root, "ArchT_raw")
    flp_created = []
    for group in ["D1", "A2A"]:
        print(group)
        group_folder = oj(root, group)
        for af in os.listdir(group_folder):
            if af.startswith("RRM") and af in animals:
                animal = af
                animal_folder = oj(group_folder, af)
                all_folders = [animal_folder]
                if fp:
                    fp_folder = oj(animal_folder, "photometry")
                    flipped = oj(animal_folder, r"LR flipped", "photometry")
                    all_folders.append(fp_folder)
                    all_folders.append(flipped)
                # TODO: later make method that save files in different folders
                for src_fd in all_folders:
                    if not os.path.exists(src_fd):
                        continue
                    for sf in os.listdir(src_fd):
                        match = None
                        for ftype in name_map:
                            mt = re.match(name_map[ftype], sf)
                            if mt:
                                match = mt
                                break
                        if match:
                            session = match.groupdict()["D"]
                            session_out = oj(out, animal, session)
                            if not os.path.exists(session_out):
                                os.makedirs(session_out)
                            target_file = oj(session_out, sf)
                            if os.path.exists(target_file):
                                continue
                            else:
                                print(
                                    f"Copying {animal} {session} {sf} to {session_out}"
                                )
                                shutil.copyfile(oj(src_fd, sf), oj(session_out, sf))
                                if "flipped" in src_fd:
                                    fname = oj(session_out, ".flp")
                                    if not os.path.exists(fname):
                                        with open(fname, "w+") as wf:
                                            print("creating .flp")
                                            flp_created.append((animal, session))

    return flp_created


def organize_RR_local(root, out, category):
    name_map = {
        "behavior": r"^RR_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "FP": r"^FP_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "FPTS": r"^FPTS_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "video": r"^RR_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
        "vidTS": r"^RR_(?P<D>Day\d+)_.*_ID-(?P<A>RRM\d+)_.*.csv",
    }

    pass


def RR_organize():
    ROOT = r"Z:\Restaurant Row\Data"
    out = r"D:\U19\data\RR\ArchT_raw"  # r"D:\U19\data\RR"
    organize_RR_structures(ROOT, out)


from neurobehavior_base import RR_Expr
from utils_rr.utils_videos import find_vid_folder_animal
from loaders.videos import RRVideoLoader
from utils_rr.utils_pose import generate_behavior_video_with_neural
from os.path import join as oj
from multiprocessing import Pool
import psutil
import tqdm


def generate_neural_with_videos_per_session(
    animal, session, rse, vidFolder, saveFolder
):
    # vidFolder = find_vid_folder_animal(vid_root, animal)
    # saveFolder = oj(pose_root, animal)
    # if not os.path.exists(saveFolder):
    #     os.makedirs(saveFolder)
    try:
        vld = RRVideoLoader(vidFolder, animal, session, outpath=saveFolder)
        bmat, neuro_series = rse.load_animal_session(animal, session)
        bdf = bmat.todf()
        vld.realign_time(bmat)
        generate_behavior_video_with_neural(
            bdf, neuro_series, vld, label_t=0.3, label=True, overwrite=True
        )
        print(animal, session, "Done")
    except:
        print(f"Error with video processing {animal} {session}")


def generate_neural_videos_all(data_root, vid_root, pose_root):
    rse = RR_Expr(data_root)
    sessions = {
        "RRM026": {151: 2, 160: 2, 167: 2, 172: 2},
        "RRM027": {155: 1, 170: 2, 175: 3},
        "RRM028": {123: 2, 130: 2, 136: 1, 141: 3, 151: 3, 156: 3},
        "RRM029": {125: 2, 130: 2, 141: 2, 153: 2, 158: 3},
        "RRM030": {139: 3, 143: 3, 146: 3, 149: 2, 154: 3, 159: 3},
        "RRM031": {125: 2, 130: 1, 134: 3, 139: 1, 143: 3, 146: 3, 149: 2},
        "RRM032": {118: 1, 122: 3, 128: 3, 132: 1, 135: 3, 138: 2, 143: 3, 147: 3},
        "RRM033": {118: 1, 122: 2, 132: 2, 135: 2, 138: 2, 143: 3, 147: 3},
        "RRM035": {195: 1, 198: 1},
        "RRM036": {161: 1, 169: 1, 172: 1, 176: 3},
    }

    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        results = []

        for animal in sessions:
            vidFolder = find_vid_folder_animal(vid_root, animal)
            saveFolder = oj(pose_root, animal)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            for sessionN in sessions[animal]:
                session = f"Day{sessionN}"
                r = pool.apply_async(
                    generate_neural_with_videos_per_session,
                    args=(animal, session, rse, vidFolder, saveFolder),
                )
                results.append(r)

        for r in tqdm.tqdm(results):
            _ = r.get()


def clean_file_with_keywords(keyword, filepaths=None):
    if filepaths is None:
        filepaths = [os.path.join(*keyword.split(os.sep)[:-1])]
        keyword = keyword.split(os.sep)[-1]
    for fpath in filepaths:
        for f in os.listdir(fpath):
            targetf = os.path.join(fpath, f)
            tdelta = datetime.now() - datetime.fromtimestamp(os.path.getctime(targetf))
            if (keyword in f) and (tdelta.seconds < 300):
                print("Deleting", targetf)
                os.remove(targetf)


def clean_bonsai_artifacts():
    clean_file_with_keywords(sys.argv[1], ["rr_data", "rr_data_FP", "rr_video"])



"""
Functions to find PSW sessions and store in dataframe
"""
def find_psw_fp_sessions(root, pattern):
    results = []
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root, f)):
            rs = find_psw_fp_sessions(os.path.join(root, f))
            results += rs
        elif f.endswith('.csv') and ('_FP_' in f):
            m = re.match(pattern, f)
            if m is None:
                print('anomaly', f)
            else:
                results.append([m.group(1), m.group(2), m.group(3), m.group(4)])
    return results

def psw_fp_session_df(root, pattern, out=None):
    psw_fp_sessions = find_psw_fp_sessions(root, pattern)
    fp_df = pd.DataFrame(psw_fp_sessions, columns=['animal', 'date', 'age', 'plug_in'])
    fp_df['date'] = pd.to_datetime(fp_df['date'], format='%Y-%m-%d')
    if out is not None:
        fp_df.to_csv(out, index=False)
    return fp_df
    


if __name__ == "__main__":
    # vid_root = r"Z:\Restaurant Row\Data"
    # data_root = r"D:\U19\data\RR\ARJ_raw"
    # pose_root = r"Z:\Restaurant Row\Data\labeled_video_neural"
    # print("running all sessions")
    # generate_neural_videos_all(data_root, vid_root, pose_root)
    track_root = r"Z:\Restaurant Row\Data\processed_tracks"
    for folder in os.listdir(track_root):
        animal_folder = os.path.join(track_root, folder)
        if os.path.isdir(animal_folder):
            for session in os.listdir(animal_folder):
                session_folder = os.path.join(animal_folder, session)
                if os.path.isdir(session_folder):
                    for f in os.listdir(session_folder):
                        if f.endswith(".csv"):
                            newf = oj(
                                session_folder,
                                f.replace("_processed", "_tracks_processed"),
                            )
                            oldf = oj(session_folder, f)
                            print(oldf, "->", newf)
                            os.rename(oldf, newf)
