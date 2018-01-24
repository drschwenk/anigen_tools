import numpy as np
import cv2
import os
import PIL.Image as pil
from skimage.measure import compare_ssim
from itertools import combinations


from subprocess import call

# trajectories_dir = 'retrieved/trajectories'
trajectories_dir = '/Users/schwenk/wrk/animation_gan/dataset/v3p0/trajectories/'
tracking_dir = 'improved_tracking'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
segmentation_dir = 'improved_segmentation'
bg_dir = 'background'
cutout_dir = 'cutout'
viz_dir = 'viz'
t_dir = trajectories_dir

kernel = np.ones((6, 6), np.uint8)


def inpaint_bg(video, frame_numbers=(10, 40, 70)):
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
    base_cv_path = '/Users/schwenk/wrk/animation_gan/build_dataset/background_creation/inpaint/'
    # base_cv_path = '/home/ubuntu/inpaint/'
    pencv_call = ["{}".format(base_cv_path + 'pencv'), "{}".format(frame_path),
                  "{}".format(cutout_path)]
    return call(pencv_call), pencv_call


def make_cutout(video, frame_arr_data, ent_masks, frame_n, write=True):
    outfile = os.path.join(trajectories_dir, cutout_dir, video.gid() + '_f{}_cutout.png'.format(frame_n))
    frame_outfile = os.path.join(trajectories_dir, cutout_dir, video.gid() + '_f{}_frame.png'.format(frame_n))
    ent_mask = cv2.dilate(ent_masks[frame_n], kernel, iterations=3)
    ent_masks = np.expand_dims(1 - ent_mask, 2)
    cutout_bg = frame_arr_data[frame_n] * ent_masks + (1 - ent_masks) * 255
    if not write:
        return cutout_bg
    pil.fromarray(cutout_bg).save(outfile)
    pil.fromarray(frame_arr_data[frame_n]).save(frame_outfile)
    return outfile, frame_outfile


def combine_masks(video):
    ent_masks = []
    for ent in video.data()['characters'] + video.data()['objects']:
        try:
            ent_mask = np.load(os.path.join(t_dir, segmentation_dir, ent.gid() + '_segm.npy.npz'))['arr_0']
            ent_masks.append(ent_mask)
        except FileNotFoundError:
            pass
    return np.logical_or.reduce(ent_masks).astype(np.uint8)


def detect_moving_bg(video):
    frame_arr_data = np.load(os.path.join(trajectories_dir, frame_arr_dir, video.gid() + '.npy'))
    ent_masks = combine_masks(video)
    vid_cutouts = np.array([make_cutout(video, frame_arr_data, ent_masks, fn, write=False) for fn in range(75)])
    avg_cutout = np.median(vid_cutouts, axis=0).astype(np.uint8)
    similarity_to_mean = np.array([compare_ssim(vid_cutouts[fn], avg_cutout, multichannel=True) for fn in range(75)])
    return similarity_to_mean.max()


def classify_video_movement(video, thresh=0.95):
    try:
        max_ssim = detect_moving_bg(video)
    except:
        return {video.gid(): 100}
    return {video.gid(): max_ssim}


def write_static_bg_frame_arr(video):
    frame_numbers = [10, 40, 70]
    bg_fill_dir = '/Users/schwenk/wrk/animation_gan/dataset/v3p0/filled_backgrounds/'
    bg_npy_dir = 'test_npz'
    three_file_paths = [os.path.join(bg_fill_dir, video.gid() + '_f{}.png'.format(fn)) for fn in frame_numbers]
    three_frames = [np.array(pil.open(bg)) for bg in three_file_paths]

    pw_combos = list(combinations(three_frames, 2))
    pw_sims = [compare_ssim(*pwc, multichannel=True) for pwc in pw_combos]

    f10_sims = sum(pw_sims[:2])
    f40_sims = sum([pw_sims[0], pw_sims[-1]])
    f70_sims = sum(pw_sims[1:])

    comp_arr = np.array([f10_sims, f40_sims, f70_sims])

    best_frame_idx = np.argmax(comp_arr)
    best_frame = three_frames[best_frame_idx]
    print(frame_numbers[best_frame_idx])
    filled_out_arr = np.tile(np.expand_dims(best_frame, 0), [75, 1, 1, 1])
    outfile = os.path.join(bg_npy_dir, video.gid() + '_bg.npy.npz')
    np.savez_compressed(outfile, np.array(filled_out_arr))
