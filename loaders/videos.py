import imageio, cv2
import tqdm
import os
import re
from behaviors import BehaviorMat
import numpy as np
import pandas as pd
import numbers


class VideoLoader:
    """Loads behavior video for a particular animal behavioral session. Allows multiple video loading

    need a video template and a timestamp template

    align_and_play: align all videos to reference, and save/play sample videos
    load_pose: load pose data, and return as posedf (for all the ROIs, search for pose, then get position)

    loading procedure:
    1. loops through folder to check for template matching
    2. put in Dict<str, (video, vidTS)>
        2.1 check template first, and then check video/vidTS (filetype), cameraID (if there are multiple cameras)
        2.2 one pass of template check that yields filetype cameraID!
    3. align video frames to timestamps (need a datastructure that stores both timestamps and videoframes)
        3.1 {camID: {'vid': VideoObj, 'vidTS': pd.Timestamp}},
        3.2 create VideoSeries Object: {'vid': VideoObj, 'vidTS': pd.Timestamp}, videoObj can handle aligning, pose estimation
    """

    def __init__(self, folder, animal, session, outpath=None) -> None:
        self.folder = folder
        self.animal = animal
        self.session = session
        self.sources = {"roi": {}}
        if outpath is None:
            outpath = folder
        self.outpath = outpath
        self.template = (
            rf"^{self.animal}_(?P<task>\w+)_{self.session}"
            + r"_(?P<fType>video|vidTS)_(?P<time>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}).(?P<ext>[a-z]{3})"
        )
        self._load()

    def _load(self):
        # could improve runtime by looping through all files in the folder in one pass and construct all the video objects
        video_template = self.template
        for f in os.listdir(self.folder):
            m = re.match(video_template, f)
            if m:
                d = m.groupdict()
                fname = os.path.join(self.folder, f)
                if d["fType"] == "video":
                    vidObj = imageio.get_reader(fname)
                    self.sources["roi"]["vid"] = vidObj
                    self.sources["roi"]["fname"] = fname
                elif d["fType"] == "vidTS":
                    vidTS = pd.read_csv(fname, names=["time"])
                    vidTS["idx"] = np.arange(len(vidTS))
                    self.sources["roi"]["vidTS"] = vidTS

    def load_pose(self, folder):
        """Takes in pose data folder and loads all the pose data and return a pose dataframe"""
        raise NotImplementedError

    def realign_time(self, reference=None):
        # Realigns timestamps to reference structure (BehaviorMat)
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)
        else:
            if reference is None:
                zero = min(self.sources[src]["vidTS"].time for src in self.sources)
            else:
                zero = reference
            assert isinstance(
                zero, numbers.Number
            ), f"reference has to be BehaviorMat, number, or None but found {type(reference)}"
            transform_func = lambda ts: ts - zero
        for src in self.sources:
            self.sources[src]["vidTS"]["time"] = transform_func(
                self.sources[src]["vidTS"]["time"]
            )

    def align_and_save(self, t, pre, post):
        """align timestamps all video snippets to t, and returns chunk of video
        # currently do not stitch videos
        pre: in seconds
        post: in seconds
        """
        t_start = t - pre
        t_end = t + post
        for src in self.sources:
            cap = cv2.VideoCapture(self.sources[src]["fname"])
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            vidObj = self.sources[src]["vid"]
            w = imageio.get_writer(
                os.path.join(
                    self.outpath,
                    f"{self.animal}_{self.session}_{src}_t{t}_s{t_start}_e{t_end}.avi",
                ),
                format="FFMPEG",
                mode="I",
                fps=fps,
            )
            tsdf = self.sources[src]["vidTS"]
            inds = tsdf.loc[
                (tsdf["time"] >= t_start) & (tsdf["time"] <= t_end), "idx"
            ].values
            for i in tqdm.tqdm(inds):
                iframe = vidObj.get_data(i)
                itime = np.around(tsdf["time"].iat[i], 2)
                yLen, xLen = iframe.shape[:2]
                piframe = cv2.putText(
                    iframe,
                    f"{src} t:{itime}",
                    (int(xLen * 0.05), yLen - int(yLen * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                w.append_data(piframe)
            w.close()
            del w

    def close(self):
        for src in self.sources:
            self.sources[src]["vid"].close()


class RRVideoLoader(VideoLoader):

    _template = r"""~ROI~_(?P<fType>cam|vidTS)_~SESS~_epoch-(?P<epoch>\d)_ID-~ANIMAL~_(?P<time>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}).(?P<ext>[a-z]{3})"""
    vid_str = "cam"

    def __init__(self, folder, animal, session, outpath=None) -> None:
        # super().__init__(folder, animal, session, outpath)
        self.folder = folder
        self.animal = animal
        self.session = session
        self.sources = {"R1": {}, "R2": {}, "R3": {}, "R4": {}}
        if outpath is None:
            outpath = folder
        self.outpath = outpath
        self.template = self._template.replace(r"~ANIMAL~", self.animal).replace(
            r"~SESS~", self.session
        )
        for roi in self.sources:
            self._load_roi(roi)

    def _load_roi(self, roi):
        if r"~ROI~" in self.template:
            video_template = self.template.replace(r"~ROI~", roi)
        else:
            video_template = self.template
        for f in os.listdir(self.folder):
            m = re.match(video_template, f)
            if m:
                d = m.groupdict()
                fname = os.path.join(self.folder, f)
                if d["fType"] == self.vid_str:
                    vidObj = imageio.get_reader(fname)
                    self.sources[roi]["vid"] = vidObj
                    self.sources[roi]["fname"] = fname
                elif d["fType"] == "vidTS":
                    vidTS = pd.read_csv(fname, names=["time"])
                    vidTS["idx"] = np.arange(len(vidTS))
                    self.sources[roi]["vidTS"] = vidTS
