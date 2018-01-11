from __future__ import division
import numpy as np
import cv2
import os
from copy import deepcopy
from skimage.segmentation import slic
from .bboxes import limit_rect
from skimage.future import graph
import pandas as pd
import PIL.Image as pil

trajectories_dir = 'trajectories'
tracking_dir = 'tracking_stabilized'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
segmentation_dir = 'improved_segmentation'
viz_dir = 'viz'

n_super_pixels = 800
n_grabcut_iter = 1
region_merge_thresh = 0.5

# slic paramas
sigma = 1
multichannel = True
convert2lab = True
slic_zero = True
enforce_connectivity = False


# hier merge params
hier_thresh = 10
rag_copy = False
in_place_merge = True

# trajectories_dir = 'trajectories'
# tracking_dir = 'tracking'
# interp_dir = 'interpolation'
# frame_arr_dir = 'frame_arr_data'
# segmentation_dir = 'segmentation'
# viz_dir = 'viz'

new_dim = 128
owidth = 640
oheight = 480

scale_down = new_dim / owidth
asp_ratio = owidth / oheight

kernel = np.ones((5, 5), np.uint8)
closing_kernel = np.ones((2, 2), np.uint8)


def mask_out_bg(img, entities, fn=0):
    img = np.array(img)[:, :, ::]
    mask = np.zeros(img.shape[:2], np.uint8)
    for ent in entities:
        rect = ent.data()['rectangles'][fn]
        x, y, x2, y2 = rect
        mask[y:y2, x:x2] = 1
    return mask


def segment_entity_rect(img, rect):
    kernel = np.ones((5, 5), np.uint8)
    img = np.array(img)[:, :, ::]
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(lab_img, mask, tuple(rect), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    cleanup_mask = np.zeros(img.shape[:2], np.uint8)
    x, y, x2, y2 = rect
    cleanup_mask[y:y2, x:x2] = 1
    final_mask = mask2[:, :, np.newaxis] * cleanup_mask[:, :, np.newaxis]
    dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)
    return dilated_mask.reshape(final_mask.shape[:2])


def scale_box(bbox):
    bb = deepcopy(bbox).reshape(2, 2)
    bb[:, 0] = bb[:, 0] * scale_down
    bb[:, 1] = bb[:, 1] * scale_down * asp_ratio
    return bb.reshape(4,)


def segment_video(video):
    t_dir = trajectories_dir
    seg_path = os.path.join(t_dir, segmentation_dir)
    frame_arr_data = np.load(os.path.join(t_dir, frame_arr_dir, video.gid() + '.npy'))
    try:
        for ent in video.data()['characters'] + video.data()['objects']:
            char_mask = []
            outfile = os.path.join(seg_path, ent.gid() + '_segm.npy')
            entity_rects = np.load(os.path.join(t_dir, tracking_dir, ent.gid() + '.npy'))
            try:
                for frame_n in range(frame_arr_data.shape[0]):
                    scaled_ent_box = scale_box(entity_rects[frame_n])
                    segm = segment_entity(frame_arr_data[frame_n], scaled_ent_box)
                    char_mask.append(segm)
            except FileNotFoundError:
                char_mask = np.zeros(frame_arr_data.shape[:3], np.uint8)
            try:
                stabilized_masks = stabilize_segmentations(char_mask)
                # stabilized_masks = char_mask
                np.savez_compressed(outfile, np.array(stabilized_masks))
            except FileNotFoundError as e:
                print(e)
    except FileExistsError:
        print(video.gid())


def stabilize_segmentations(char_mask):
    char_mask = np.array(char_mask)
    ent_area = char_mask.sum(axis=1).sum(axis=1).astype(np.float64)
    # low_area_thresh = np.percentile(ent_area, 30)
    # high_area_thresh = np.percentile(ent_area, 90)
    med_area = np.median(ent_area)
    low_area_thresh = med_area * 0.8
    high_area_thresh = med_area * 1.1
    good_frames = (high_area_thresh >= ent_area) & (ent_area >= low_area_thresh)
    first_good_frame = good_frames.argmax()
    patched_ent_masks = np.zeros_like(char_mask)
    for fn in range(0, char_mask.shape[0]):
        if not good_frames[fn]:
            if fn > first_good_frame:
                patched_ent_masks[fn] = patched_ent_masks[fn - 1]
            else:
                patched_ent_masks[fn] = char_mask[first_good_frame]
        else:
            if abs(ent_area[fn] - ent_area[fn - 1]) / ent_area[fn] > 0.03 and fn > 0:
                patched_ent_masks[fn] = patched_ent_masks[fn - 1]
            else:
                patched_ent_masks[fn] = char_mask[fn]
    return patched_ent_masks


def draw_segmentation(segmentation_arr):
    return pil.fromarray(segmentation_arr).convert('P', dither=None, palette=pil.ADAPTIVE)


def degrade_colorspace(image):
    return


def weight_mean_color(graph, src, dst, n):
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def compute_iou(segment1, segment2, area1=0, area2=0):
    area_intersection = np.sum(segment1 & segment2)
    if area_intersection == 0:
        return 0, 0, 0
    if area1 == 0:
        area1 = np.sum(segment1)
    if area2 == 0:
        area2 = np.sum(segment2)
    area_overlap = area_intersection / (area1 + area2 - area_intersection + 0.01)
    area_self_overlap1 = area_intersection / (area1 + 0.01)
    area_self_overlap2 = area_intersection / (area2 + 0.01)
    return area_overlap, area_self_overlap1, area_self_overlap2


def create_bbox_segment(rect, frame_size=(128, 128)):
    segment = np.zeros(frame_size, dtype=np.uint8)
    segment[rect[0]: rect[2], rect[1]: rect[3]] = 1
    return segment.T


def partition_image(img, n_segments):
    # lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    superpixels = slic(img, n_segments, sigma=sigma, multichannel=multichannel, convert2lab=convert2lab,
                       slic_zero=slic_zero, enforce_connectivity=False)
    ent_rag = graph.rag_mean_color(img, superpixels)
    regions = graph.merge_hierarchical(superpixels, ent_rag, thresh=hier_thresh, rag_copy=rag_copy,
                                       in_place_merge=in_place_merge,
                                       merge_func=merge_mean_color,
                                       weight_func=weight_mean_color)
    #     out = color.label2rgb(regions, img, kind='avg')
    #     out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    return regions


def rough_segment(regions, bbox_mask, inclusion_thresh):
    ent_reg_overlaps = []
    for reg in np.unique(regions):
        reg_mask = regions == reg
        reg_mask = reg_mask.astype(np.uint8)
        reg_iou = compute_iou(bbox_mask, reg_mask)[2]
        ent_reg_overlaps.append(reg_iou)
    overlapping_regions = pd.Series(ent_reg_overlaps).sort_values(ascending=False)
    regions_to_include = overlapping_regions.index[overlapping_regions > inclusion_thresh]
    ent_segment = np.isin(regions, regions_to_include).astype(np.uint8)
    return ent_segment


def grabcut_from_rough_mask(ent_mask, img, bg_mask, fg_mask):
    mask = np.where(ent_mask == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    mask *= bg_mask     # sets certain bg outside enlarged bounding box
    mask = mask - fg_mask * 2
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgm, fgm, n_grabcut_iter, cv2.GC_INIT_WITH_MASK)
    ref_mask = np.where((mask == 3) | (mask == 1), 1, 0).astype('uint8')
    return ref_mask


def segment_entity(frame, ent_rect, grabcut_failure_thresh = 0.35):
    img = deepcopy(frame)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ent_rect = limit_rect(np.array(ent_rect).reshape(2, 2), new_dim, new_dim, 0).reshape(4)
    ent_bbox_mask = create_bbox_segment(ent_rect)
    inv_bg_mask = cv2.dilate(ent_bbox_mask, kernel, iterations=2)
    img_within_bbox = img * np.tile(np.expand_dims(ent_bbox_mask, 2), [1, 1, 3])
    gray = cv2.cvtColor(img_within_bbox, cv2.COLOR_RGB2GRAY)
    def_fg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    def_fg = ((def_fg == 0) * ent_bbox_mask).astype(np.uint8)
    img_regions = partition_image(lab, n_super_pixels)
    rough_ent = rough_segment(img_regions, ent_bbox_mask, region_merge_thresh)
    if not rough_ent.sum():
        rough_ent = ent_bbox_mask
    ent_segmentation = grabcut_from_rough_mask(rough_ent, lab, inv_bg_mask, def_fg)
    if compute_iou(ent_segmentation, ent_bbox_mask)[0] < grabcut_failure_thresh:
        ent_segmentation = rough_ent
        ent_segmentation = cv2.morphologyEx(ent_segmentation, cv2.MORPH_OPEN, kernel)
    else:
        ent_segmentation = cv2.morphologyEx(ent_segmentation, cv2.MORPH_OPEN, closing_kernel)
    return ent_segmentation


def get_vid_frame_data(vid_gid):
    return np.load(os.path.join(trajectories_dir, frame_arr_dir, vid_gid + '.npy'))


def get_ent_tracking(ent):
    return np.load('./trajectories/tracking/' + ent.gid() + '.npy')


def gen_single_segmentation(video, ent, frame_n=30):
    test_arr_img = get_vid_frame_data(video.gid())
    anim_frame = test_arr_img[frame_n]
    ent_rects = get_ent_tracking(ent)
    ent_rects = ent_rects[frame_n]
    scaled_ent_box = scale_box(ent_rects)
    ent_segmentation = segment_entity(anim_frame, scaled_ent_box)
    rgb_ent_segmentation = np.tile(np.expand_dims(ent_segmentation, 2), [1, 1, 3])
    inv_mask = np.logical_not(rgb_ent_segmentation).astype(np.uint8)
    cutout_ent = anim_frame * rgb_ent_segmentation + inv_mask * 90
    return pil.fromarray(cutout_ent), ent_segmentation


def draw_video_segmentations(video, frame_arr_data=np.array([]), retrieved=False):
    if retrieved:
        t_dir = './retrieved/' + trajectories_dir
    else:
        t_dir = trajectories_dir
    # try:
    seg_path = os.path.join(t_dir, viz_dir)
    if not frame_arr_data.any():
        frame_arr_data = np.load(os.path.join(t_dir,  frame_arr_dir, video.gid() + '.npy'))

        for ent in video.data()['characters'] + video.data()['objects']:
            try:
                outfile = os.path.join(seg_path, ent.gid() + '_segm.gif').replace(' ', '_')
                char_mask = np.load(os.path.join(t_dir,  segmentation_dir, ent.gid() + '_segm.npy.npz'))
                char_mask = np.expand_dims(char_mask['arr_0'], 3)
                inv_mask = np.logical_not(char_mask).astype(np.uint8)
                segm_arr = frame_arr_data * char_mask + inv_mask * 90
                segmentation_frames = [draw_segmentation(segm_arr[frame_n]) for frame_n in
                                       range(segm_arr.shape[0])]
                segmentation_frames[0].save(outfile, save_all=True, optimize=False, duration=42,
                                            append_images=segmentation_frames[1:])
            except FileNotFoundError:
                pass
