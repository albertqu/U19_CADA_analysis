import pandas as pd
import os
import numpy as np
import cv2
import imageio
import tqdm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import warnings


def preprocess_turn_df(bdf):
    """Function to preprocess behavior df for turn analysis"""
    turn_df = bdf[
        [
            "restaurant",
            "lapIndex",
            "trial",
            "tone_onset",
            "offer_prob",
            "T_Entry",
            "exit",
            "quit",
            "accept",
            "choice",
            "outcome",
            "collection",
            "quit_time",
        ]
    ].reset_index(drop=True)
    turn_df["T_Entry{t+1}"] = turn_df["T_Entry"].shift(-1)
    turn_df["turn_end"] = turn_df["T_Entry{t+1}"]
    tensel = turn_df["T_Entry{t+1}"].isnull()
    turn_df.loc[tensel, "turn_end"] = turn_df.loc[tensel, "T_Entry"] + 1
    casel = (turn_df["accept"] == 1) & (turn_df["quit"].isnull())
    turn_df.loc[casel, "turn_end"] = turn_df.loc[casel, "choice"] + 1
    turn_df["duration"] = turn_df["turn_end"] - turn_df["tone_onset"]
    turn_df["AR"] = turn_df["accept"].map({1: "ACC", 0: "REJ"})
    turn_df["offer_tone"] = turn_df["offer_prob"].apply(lambda x: f"{int(x)}tone")
    turn_df["ACC"] = np.nan
    turn_df.loc[turn_df["accept"] == 1, "ACC"] = turn_df.loc[
        turn_df["accept"] == 1, "choice"
    ]
    turn_df["REJ"] = np.nan
    turn_df.loc[turn_df["accept"] == 0, "REJ"] = turn_df.loc[
        turn_df["accept"] == 0, "choice"
    ]
    turn_df.drop(turn_df[turn_df["quit_time"] > 7].index, inplace=True)
    return turn_df


def get_turn_videos(bdf, vld, r_num, label_t=0.2, label=True, overwrite=False):
    """Function to separate experiment videos into mini-turn videos, and print specific events:
    r_num: restaurant number
    bdf: behavior dataframe
    vld: video loader
    event_cols: default= [tone_onset, T_Entry, ACC, REJ, quit]

    TODO: Do we analyze
    1. mistriggered reject/quit trials where animals wait at reward zone
    2. mistriggered trials where animal wait after T Entry and has a long putative quit

    """
    turn_df = preprocess_turn_df(bdf)

    event_cols = ["tone_onset", "T_Entry", "choice", "quit"]
    event_prop = {"choice": "AR", "tone_onset": "offer_tone"}

    i = r_num
    roi = f"R{i}"
    ts_df = vld.sources[roi]["vidTS"]
    ir_df = turn_df[turn_df["restaurant"] == i].reset_index(drop=True)

    jtdfs = []

    for j in range(len(ir_df)):
        tstart, tend = ir_df.loc[j, "tone_onset"], ir_df.loc[j, "turn_end"]
        jt_df = ts_df[(ts_df["time"] >= tstart) & (ts_df["time"] <= tend)].reset_index(
            drop=True
        )
        jt_df["label"] = ""
        jt_df["rel_time"] = jt_df["time"] - tstart
        for k, ec in enumerate(event_cols):
            etime = ir_df.loc[j, ec]
            if ec in event_prop:
                ec_label = ir_df.loc[j, event_prop[ec]]
            else:
                ec_label = ec
            if not pd.isnull(etime):
                jt_df.loc[
                    (jt_df["time"] >= etime) & (jt_df["time"] <= etime + label_t),
                    "label",
                ] = ec_label
        # print(j, ir_df.iloc[j, 1:4])
        jt_df[["restaurant", "lapIndex", "trial"]] = ir_df.iloc[j, 0:3].values
        # vld.close()
        jtdfs.append(jt_df)
    frame_df = pd.concat(jtdfs, axis=0).reset_index(drop=True)

    vfname = os.path.join(vld.outpath, f"{vld.animal}_{vld.session}_{roi}_turns.avi")
    if (not os.path.exists(vfname)) or overwrite:
        cap = cv2.VideoCapture(vld.sources[roi]["fname"])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        vidObj = vld.sources[roi]["vid"]
        w = imageio.get_writer(
            vfname,
            format="FFMPEG",
            mode="I",
            fps=fps,
        )
        for i in tqdm.tqdm(range(len(frame_df))):
            iframe = vidObj.get_data(frame_df.loc[i, "idx"])
            itime = np.around(frame_df["time"].iat[i], 3)
            irtime = np.around(frame_df["rel_time"].iat[i], 3)
            itrial = frame_df["trial"].iat[i].astype(int)
            ilabel = frame_df["label"].iat[i]
            yLen, xLen = iframe.shape[:2]

            if label:
                text = f"T{itrial} t:{itime} rel:{irtime}"
                piframe = cv2.putText(
                    iframe,
                    text,
                    (int(xLen * 0.05), yLen - int(yLen * 0.23)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 240, 0),
                    1,
                )
                if ilabel:
                    piframe = cv2.putText(
                        iframe,
                        ilabel,
                        (int(xLen * 0.05), yLen - int(yLen * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 240, 0),
                        1,
                    )
            else:
                piframe = iframe
            w.append_data(piframe)
        w.close()
        del w
    return frame_df


def get_turn_frame_df(turn_df, vld, r_num, label_t=0.2):
    # processing frame information
    event_cols = ["tone_onset", "T_Entry", "choice", "quit"]
    event_prop = {"choice": "AR", "tone_onset": "offer_tone"}

    i = r_num
    roi = f"R{i}"
    ts_df = vld.sources[roi]["vidTS"]
    ir_df = turn_df[turn_df["restaurant"] == i].reset_index(drop=True)

    jtdfs = []

    for j in range(len(ir_df)):
        tstart, tend = ir_df.loc[j, "tone_onset"], ir_df.loc[j, "turn_end"]
        jt_df = ts_df[(ts_df["time"] >= tstart) & (ts_df["time"] <= tend)].reset_index(
            drop=True
        )
        jt_df["label"] = ""
        jt_df["rel_time"] = jt_df["time"] - tstart
        jt_df["start"] = tstart
        jt_df["end"] = tend
        for k, ec in enumerate(event_cols):
            etime = ir_df.loc[j, ec]
            if ec in event_prop:
                ec_label = ir_df.loc[j, event_prop[ec]]
            else:
                ec_label = ec
            if not pd.isnull(etime):
                jt_df.loc[
                    (jt_df["time"] >= etime) & (jt_df["time"] <= etime + label_t),
                    "label",
                ] = ec_label
        # print(j, ir_df.iloc[j, 1:4])
        jt_df[["restaurant", "lapIndex", "trial"]] = ir_df.iloc[j, 0:3].values
        # vld.close()
        jtdfs.append(jt_df)
    frame_df = pd.concat(jtdfs, axis=0).reset_index(drop=True)
    return frame_df


def add_neural_to_frames(frame_df, neuro_series, vld, roi, label=True, overwrite=True):
    # setting up artists
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 2, hspace=0.3, figure=fig)
    ax0 = fig.add_subplot(gs[:2, 0])
    ax0.set_axis_off()
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    axesH = [ax1, ax2]
    axesL = np.zeros((2, 2), dtype=object)
    for j in range(2):
        for i in range(2):
            if j == 1:
                axesL[i, j] = fig.add_subplot(gs[i + 2, j], sharex=ax1)
            elif i == 1:
                axesL[i, j] = fig.add_subplot(gs[i + 2, j], sharex=axesL[i - 1, j])
            else:
                axesL[i, j] = fig.add_subplot(gs[i + 2, j])
    sns.despine()

    # setting up dataloaders
    cap = cv2.VideoCapture(vld.sources[roi]["fname"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    vidObj = vld.sources[roi]["vid"]
    raw_df = neuro_series.neural_df
    dff_df = neuro_series.calculate_dff(method="lossless")
    t_between = lambda s, t0, t1: (s >= t0) & (s < t1)
    prev_trial = 0

    def animate_img_frame(i):
        for ax in fig.axes:
            ax.cla()  # clear the previous image
        iframe = vidObj.get_data(frame_df.loc[i, "idx"])
        itime = np.around(frame_df["time"].iat[i], 3)
        irtime = np.around(frame_df["rel_time"].iat[i], 3)
        itrial = frame_df["trial"].iat[i].astype(int)
        ilabel = frame_df["label"].iat[i]
        tstart = frame_df["start"].iat[i]
        tend = frame_df["end"].iat[i]
        t = itime
        yLen, xLen = iframe.shape[:2]

        if label:
            text = f"T{itrial} t:{itime} rel:{irtime}"
            piframe = cv2.putText(
                iframe,
                text,
                (int(xLen * 0.05), yLen - int(yLen * 0.23)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 240, 0),
                1,
            )
            if ilabel:
                piframe = cv2.putText(
                    iframe,
                    ilabel,
                    (int(xLen * 0.05), yLen - int(yLen * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 240, 0),
                    1,
                )

        ax0.imshow(piframe)
        side_styles = ["-", "--"]
        for j, ch in enumerate(["left_470nm", "right_470nm"]):
            ch_df = dff_df[dff_df["roi"] == ch]
            if not ch_df.empty:
                sns.lineplot(
                    ch_df[t_between(ch_df["time"], tstart, tend)],
                    x="time",
                    y="ZdFF",
                    ax=axesH[j],
                    label=ch,
                    ls=side_styles[j],
                )
                axesH[j].axvline(t, ls="-", color="k")
                axesH[j].legend()

        ch_colors = ["r", "b"]
        for ii, ch in enumerate(["470nm", "410nm"]):
            for j, side in enumerate(["left", "right"]):
                side_ch = f"{side}_{ch}"
                if side_ch in raw_df.columns:
                    sns.lineplot(
                        raw_df[t_between(raw_df["time"], tstart, tend)],
                        x="time",
                        y=side_ch,
                        ax=axesL[ii, j],
                        color=ch_colors[ii],
                        ls=side_styles[j],
                    )
                    axesL[ii, j].axvline(t, ls="-", color="k")
        # artists = []
        # for ax in fig.axes:
        #     artists += ax.lines
        #     artists += ax.collections
        #     artists += ax.images
        # return artists
        canvas = fig.canvas
        canvas.draw()  # Draw the canvas, cache the renderer
        v = canvas.renderer.buffer_rgba()
        image_flat = np.frombuffer(v, dtype="uint8")  # (H * W * 3,)
        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 4)  # (H, W, 3)
        return image

    # slow solution
    # anim = animation.FuncAnimation(fig, animate, frames = 30, interval = int(1000/fps), blit = True)
    # anim.save(fname)

    vfname = os.path.join(
        vld.outpath, f"{vld.animal}_{vld.session}_{roi}_turns_neural.avi"
    )
    warnings.filterwarnings("ignore")
    if (not os.path.exists(vfname)) or overwrite:
        w = imageio.get_writer(
            vfname,
            format="FFMPEG",
            mode="I",
            fps=fps,
        )
        for i in range(len(frame_df)):
            w.append_data(animate_img_frame(i))
        w.close()
        del w
    plt.close()
    return vfname


def generate_behavior_video_with_neural(
    bdf, neuro_series, vld, label_t=0.2, label=True, overwrite=True
):
    """Function that takes in behavior dataframe, NeuroSeries object, VideoLoader object and generate corresponding
    video overlayyed with neural data. Used as a qualitative check.
    label_t: float
        duration of event caption in seconds.
    label: bool
        whether to display event caption labels
    overwrite: bool
        whether to overwrite existing videos.
    """
    turn_df = preprocess_turn_df(bdf)
    for r_num in tqdm.tqdm(range(1, 5)):
        roi = f"R{r_num}"
        frame_df = get_turn_frame_df(turn_df, vld, r_num, label_t)
        add_neural_to_frames(frame_df, neuro_series, vld, roi, label, overwrite)
