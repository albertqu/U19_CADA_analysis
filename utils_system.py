import os, shutil, stat, errno, time, datetime, re

import pandas as pd
import numpy as np

from utils import path_prefix_free
import imageio, cv2
from os.path import join as oj



def archive_by_date(src_folder, dest_folder, archive_date):
    assert os.path.exists(dest_folder)
    os.chdir(src_folder)
    tomoves = []
    exceptions = ['_MouseBrain_Atlas3', 'Lab 4CR Digging Masterfile', 'Talks']
    for name in os.listdir(src_folder):
        if (name[0] != '_') and (datetime.datetime.fromtimestamp(os.path.getatime(name)) < archive_date):
            #atime: access time, mtime: modify time
            # print(name, datetime.datetime.fromtimestamp(os.path.getmtime(name)))
            if name not in exceptions:
                tomoves.append(name)
    for name in tomoves:
        try:
            print("moving", name)
            shutil.move(name, os.path.join(dest_folder, name))
        except Exception:
            print("skipping", name)


def chunk_video_sample_from_file(filename, out_folder, fps, duration):
    assert path_prefix_free(filename).count('.') == 1, 'has to contain only one .'
    file_code = path_prefix_free(filename).split('.')[0]
    suffix = path_prefix_free(filename).split('.')[1]
    w = imageio.get_writer(os.path.join(out_folder, f'{file_code}_sample.{suffix}'), format='FFMPEG',
                           mode='I', fps=fps) # figure out what mode means
    vid = imageio.get_reader(filename)
    for ith, img in enumerate(vid):
        if ith < duration * fps:
            print('writing', img.shape, ith)
            w.append_data(img)
        if ith >= duration * fps:
            print('all done')
            break
    w.close()


def chunk_four_vids_and_stitch(filenames, ts_names, out_folder, fps, duration, time_zero=0):
    vids = [imageio.get_reader(fname) for fname in filenames]
    tss = [(pd.read_csv(tsn, names=['time']) - time_zero) / 1000 for tsn in ts_names]
    file_code = path_prefix_free(filenames[0]).split('.')[0][3:]
    suffix = path_prefix_free(filenames[0]).split('.')[1]
    w = imageio.get_writer(os.path.join(out_folder, f'{file_code}_sample.{suffix}'), format='FFMPEG',
                           mode='I', fps=fps)  # figure out what mode means

    alt_order = {2: 0, 1: 1, 3: 2, 0: 3}
    total_frames = min(duration * fps, min(len(tsdf) for tsdf in tss))
    for i in range(total_frames):
        frames = [[None, None], [None, None]]
        for j in range(4):
            jframe = vids[j].get_data(i)
            jtime = np.around(tss[j].time[i], 2)
            pjframe = cv2.putText(jframe, f'R{j+1}: {jtime}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            r, c = alt_order[j] // 2, alt_order[j] % 2
            frames[r][c] = pjframe

        frames[0] = np.concatenate(frames[0], axis=1)
        frames[1] = np.concatenate(frames[1], axis=1)
        final_frame = np.concatenate(frames, axis=0)
        w.append_data(final_frame)
        print(f'writing {i}th')
    w.close()


def chunk_helper_session(animal, session, folder, out_folder, duration=20*60):
    fps = 30
    vid_files = [None] * 4
    vid_times = [None] * 4
    for f in os.listdir(os.path.join(folder, animal)):
        if (animal in f) and (session in f) and (f.startswith('RR_')):
            bfile = os.path.join(folder, animal, f)
            bdf = pd.read_csv(bfile, sep = ' ', header = None, names = ['time', 'b_code', 'none'])
            df_zero = bdf.loc[0, 'time']
            break

    for f in os.listdir(os.path.join(folder, animal, 'video')):
        for i in range(1, 5):
            vidf = f'R{i}_cam'
            vtsf = f'R{i}_vidTS'
            if (animal in f) and (session in f) and (f.startswith(vidf)):
                vid_files[i-1] = os.path.join(folder, animal, 'video', f)
            elif (animal in f) and (session in f) and (f.startswith(vtsf)):
                vid_times[i-1] = os.path.join(folder, animal, 'video', f)
    chunk_four_vids_and_stitch(vid_files, vid_times, out_folder, fps, duration, time_zero=df_zero)


"""####################################################
################### Data Structure ####################
####################################################"""


def rename_dir_files_recursive(root, namemap):
    for p in os.listdir(root):
        pname = os.path.join(root, p)
        if os.path.isdir(pname):
            rename_dir_files_recursive(pname, namemap)
        for name in namemap:
            if name in p:
                newpname = os.path.join(root, p.replace(name, namemap[name]))
                os.rename(pname, newpname)


def make_stim_blocks(n=20, p=0.25, zero=False):
    # p: percentage of stimulation, p<0.5
    """ Use this example to visualize:
    import matplotlib.pyplot as plt
    plt.stem(make_stim_blocks(20))
    plt.xticks(np.arange(20)[4::5], np.arange(1, 21)[4::5])
    """
    if zero:
        return np.concatenate([[0], make_stim_blocks(n, p)])
    subN = int(n * p)
    stim_seq = np.zeros(n)
    if subN:
        randInds = np.sort(np.random.choice(n-subN, subN, replace=False)) + np.arange(subN)
        stim_seq[randInds] = 1
    return stim_seq


def make_long_stim_blocks(N, p=0.25, n=20):
    all_stims = []
    loops = N // n
    tail = N % n
    zero = False
    for i in range(loops):
        stim_seq = make_stim_blocks(n, p, zero)
        zero = (stim_seq[-1] == 1)
        all_stims.append(stim_seq)
    all_stims.append(make_stim_blocks(tail, p, zero))
    return np.concatenate(all_stims)


if __name__ == '__main__':
    # archive_date = datetime.datetime.strptime("2019/05/31", "%Y/%m/%d")
    # src_folder = "/Volumes/Wilbrecht_file_server"
    # archive = os.path.join(src_folder, '_ARCHIVE')
    # if not os.path.exists(archive):
    #     os.makedirs(archive)
    # archive_by_date(src_folder, archive, archive_date)
    ROOT = r"Z:\Restaurant Row\Data"
    out = r"D:\U19\data\RR"
    organize_RR_structures(ROOT, out)

