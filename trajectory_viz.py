import cv2
import numpy as np
import PIL.Image as pil
from tqdm import tqdm
import copy
import numpy_indexed as npi
from bboxes import comp_box_center

data_dir = '/Users/schwenk/wrk/animation_gan/dataset/v2p0/trajectories'
tracking_dir = 'tracking'


def filter_chars(vid, n_chars = None, chars_required=None):
    if n_chars:
        len_match = len(vid.data()['characters']) == n_chars
    else:
        len_match = True
    if chars_required:
        chars_present = set([char.data()['entityLabel'] for char in vid.data()['characters']])
        chars_satisfied = len(set(chars_present).intersection(set(chars_required))) == len(chars_required)
    else:
        chars_satisfied = True
    return len_match and chars_satisfied


def filter_description(vid, contains=(), doesnot_contain=()):
    contains_satisfied = sum([phrase.lower() in vid.description() for phrase in contains]) == len(contains)
    doesnot_contains_satisfied = sum([phrase.lower() not in vid.description() for phrase in doesnot_contain]) == len(doesnot_contain)
    return contains_satisfied and doesnot_contains_satisfied


def vid_filter(vid, filters):
    return sum([filt(vid) for filt in filters]) == len(filters)


def avg_video_appearance(videos, viz=False):
    width, height = (640, 480)
    n_imgs = len(videos)
    avg_arr = np.zeros((height, width, 3), np.float)
    for vid in tqdm(videos):
        im = vid.get_key_frame_images(spec_frame=1).pop()
        img_arr = np.array(im, dtype=np.float)
        avg_arr += img_arr / n_imgs
    avg_arr = np.array(np.round(avg_arr), dtype=np.uint8)
    if viz:
        return pil.fromarray(avg_arr, mode="RGB")
    return avg_arr


def gen_smooth_traj(tracking_arr):
    tracked_obj_centers = np.apply_along_axis(comp_box_center, 1, tracking_arr)
    tracked_obj_centers.sort(axis=0)
    unique_positions = np.unique(tracked_obj_centers, axis=0)
    obj_x = unique_positions[:, 0]
    obj_y = unique_positions[:, 1]
    x_u, y_m = npi.group_by(obj_x).mean(obj_y)
    polyf = np.poly1d(np.polyfit(x_u, y_m, 1))
    xr = np.linspace(x_u[0], x_u[-1], num=400, endpoint=True)
    curve_pts = np.vstack((xr, polyf(xr))).T
    return curve_pts.astype(np.uint16)


def draw_trajectory(traj_img, curve, mag):
    movement_thresh = 10
    n_points = len(curve)
    if curve[-1][0] == 0 or curve[0][0] == 0:
        return
    if np.linalg.norm(curve[-1] - curve[0]) < movement_thresh:
        cv2.circle(traj_img, tuple(curve[0]), 1, (100, 0, 0), -1)
        return
    for idx, point in enumerate(curve):
        p_size = 1
        if idx == 0 or idx == n_points - 1:
            p_size = 4
        pf = idx / n_points
        p_color = (100 * pf, 100 * (1 - pf), 0)
        cv2.circle(traj_img, tuple(point), p_size, p_color, -1)


def draw_trajectory_poly(traj_img, curve, mag):
    return cv2.polylines(traj_img, np.int32([curve]), False, (mag, mag, mag), thickness=2)


def draw_trajectory_set(videos, ent_label, avg_img=None, mag=50, ent_type = 'objects'):
    if not avg_img.any():
        avg_img = avg_video_appearance(videos)
    to_draw = copy.deepcopy(avg_img)
    for vid in videos:
        try:
            target_ent = [ent for ent in vid.data()[ent_type] if ent_label in ent.data()['entityLabel']].pop()
        except IndexError:
            continue
        traj_layer = np.zeros_like(to_draw)
        try:
            tracking_arr = np.load(os.path.join(data_dir,  tracking_dir, target_ent.gid() + '.npy'))
        except FileNotFoundError:
            print('here')
            pass
        smoothed_traj = gen_smooth_traj(tracking_arr)
        draw_trajectory(traj_layer, smoothed_traj, mag)
        to_draw = cv2.addWeighted(traj_layer, mag, to_draw, 1 , 0)
    return to_draw
