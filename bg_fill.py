import numpy as np
import cv2
import os
import PIL.Image as pil
from subprocess import call

trajectories_dir = 'trajectories'
tracking_dir = 'tracking'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
segmentation_dir = 'segmentation'
bg_dir = 'background'
cutout_dir = 'cutout'
viz_dir = 'viz'
t_dir = trajectories_dir


def inpaint_bg(video, frame_numbers=(10, 40)):
    try:
        frame_arr_data = np.load(os.path.join(t_dir,  frame_arr_dir, video.gid() + '.npy'))
        ent_masks = combine_masks(video)
        for frame_n in frame_numbers:
            cutout_path, frame_path = make_cutout(video, frame_arr_data, ent_masks, frame_n)
            ret, call_made = call_patch_match(cutout_path, frame_path)
            if ret != 0:
                print(video.gid(), ret)
                return call_made
    except:
        print(video.gid())


def call_patch_match(cutout_path, frame_path):
    # base_cv_path = '/Users/schwenk/wrk/animation_gan/build_dataset/background_creation/inpaint/'
    base_cv_path = '/home/ubuntu/inpaint/'
    pencv_call = ["{}".format(base_cv_path + 'pencv'), "{}".format(frame_path),
                  "{}".format(cutout_path)]
    return call(pencv_call), pencv_call


def make_cutout(video, frame_arr_data, ent_masks, frame_n):
    outfile = os.path.join(trajectories_dir, cutout_dir, video.gid() + '_f{}_cutout.png'.format(frame_n))
    frame_outfile = os.path.join(trajectories_dir, cutout_dir, video.gid() + '_f{}_frame.png'.format(frame_n))
    ent_masks = np.expand_dims(1 - ent_masks[frame_n], 2)
    cutout_bg = frame_arr_data[frame_n] * ent_masks + (1 - ent_masks) * 255
    pil.fromarray(cutout_bg).save(outfile)
    pil.fromarray(frame_arr_data[frame_n]).save(frame_outfile)
    return outfile, frame_outfile


def combine_masks(video):
    ent_masks = []
    for ent in video.data()['characters'] + video.data()['objects']:
        ent_masks.append(np.load(os.path.join(t_dir, segmentation_dir, ent.gid() + '_segm.npy.npz'))['arr_0'])
    return np.logical_or.reduce(ent_masks).astype(np.uint8)


