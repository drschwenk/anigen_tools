from __future__ import division
import numpy as np
import cv2
import os
from copy import deepcopy
from skimage.segmentation import slic
# from skimage.segmentation import mark_boundaries
# from skimage import data, io, segmentation, color
from skimage.future import graph
import pandas as pd
import PIL.Image as pil

trajectories_dir = 'trajectories'
tracking_dir = 'tracking_stabilized'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
segmentation_dir = 'improved_segmentation'
viz_dir = 'viz'


n_super_pixels = 100

n_grabcut_iter = 1

region_merge_thresh = 0.2

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
closing_kernel = np.ones((10, 10), np.uint8)


def mask_out_bg(img, entities, fn=0):
    img = np.array(img)[:, :, ::]
    mask = np.zeros(img.shape[:2], np.uint8)
    for ent in entities:
        rect = ent.data()['rectangles'][fn]
        x, y, x2, y2 = rect
        mask[y:y2, x:x2] = 1
    return mask


# def segment_entity(img, rect, bg_mask):
#     mask = bg_mask
#     img = np.array(img)[:, :, ::]
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
#     mask, bgd_model, fgd_model = cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     img = img * mask[:, :, np.newaxis]
#     return img


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


def segment_all_video_entities(video, retrieved=False):
    if retrieved:
        t_dir = './retrieved/' + trajectories_dir
    else:
        t_dir = trajectories_dir
    seg_path = os.path.join(t_dir, segmentation_dir)
    frame_arr_data = np.load(os.path.join(t_dir,  frame_arr_dir, video.gid() + '.npy'))
    for ent in video.data()['characters'] + video.data()['objects']:
        char_mask = []
        outfile = os.path.join(seg_path, ent.gid() + '_segm.npy')
        try:
            for frame_n in range(frame_arr_data.shape[0]):
                entity_rects = np.load(os.path.join(t_dir, tracking_dir, ent.gid() + '.npy'))
                scaled_ent_box = scale_box(entity_rects[frame_n])
                char_mask.append(segment_entity_rect(frame_arr_data[frame_n], scaled_ent_box))
        except:
            char_mask = np.zeros(frame_arr_data.shape[:3], np.uint8)
        try:
            np.savez_compressed(outfile, np.array(char_mask))
        except FileNotFoundError as e:
            print(e)


def segment_video(video):
    t_dir = trajectories_dir
    seg_path = os.path.join(t_dir, segmentation_dir)
    frame_arr_data = np.load(os.path.join(t_dir, frame_arr_dir, video.gid() + '.npy'))
    try:
        for ent in video.data()['characters']: # + video.data()['objects']:
            char_mask = []
            outfile = os.path.join(seg_path, ent.gid() + '_segm.npy')
            entity_rects = np.load(os.path.join(t_dir, tracking_dir, ent.gid() + '.npy'))
            other_ents = [oe for oe in video.data()['characters'] + video.data()['objects'] if oe.gid() != ent.gid()]
            other_rects = [np.load(os.path.join(t_dir, tracking_dir, oe.gid() + '.npy')) for oe in other_ents]
            try:
                # bgm = np.zeros((1, 65), np.float64)
                # fgm = np.zeros((1, 65), np.float64)
                for frame_n in range(frame_arr_data.shape[0]):
                    bgm = np.zeros((1, 65), np.float64)
                    fgm = np.zeros((1, 65), np.float64)
                    scaled_ent_box = scale_box(entity_rects[frame_n])
                    # scaled_other = [scale_box(oer[frame_n]) for oer in other_rects]
                    segm, bgm, fgm = segment_entity(frame_arr_data[frame_n], scaled_ent_box, bgm, fgm, other_rects)
                    char_mask.append(segm)
            except FileNotFoundError:
                char_mask = np.zeros(frame_arr_data.shape[:3], np.uint8)
            try:
                np.savez_compressed(outfile, np.array(char_mask))
            except FileNotFoundError as e:
                print(e)
    except FileExistsError:
        print(ent.gid())


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
    # rect = scale_box(np.array(ent.rect())[1])
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
    #     masked_ent = frame * np.tile(np.expand_dims(ent_segment, 2), [1, 1, 3])
    return ent_segment


def grabcut_from_rough_mask(ent_mask, img, bg_mask, fg_mask, bgdModel, fgdModel):
    import pdb
    mask = np.where(ent_mask == 1, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    # print(pd.value_counts(pd.Series(mask.ravel())))
    mask *= bg_mask     # sets certain bg outside enlarged bounding box
    mask = mask - fg_mask * 2
    # print(pd.value_counts(pd.Series(mask.ravel())))
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgdModel, fgdModel, n_grabcut_iter, cv2.GC_INIT_WITH_MASK)
    ref_mask = np.where((mask == 3) | (mask == 1), 1, 0).astype('uint8')

    # ref_mask = ref_mask * ent_mask
    # if np.array(combined_other_mask).any():
    #     ref_mask = ref_mask * (combined_other_mask == 0)
    return ref_mask, bgdModel, fgdModel


def segment_entity(frame, ent_rect, bgm, fgm, other_bbox):
    img = deepcopy(frame)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ent_bbox_mask = create_bbox_segment(ent_rect)
    inv_bg_mask = cv2.dilate(ent_bbox_mask, kernel, iterations=2)
    img_within_bbox = img * np.tile(np.expand_dims(ent_bbox_mask, 2), [1, 1, 3])
    gray = cv2.cvtColor(img_within_bbox, cv2.COLOR_RGB2GRAY)
    def_fg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    def_fg = (def_fg == 0) * ent_bbox_mask
    def_fg = def_fg.astype(np.uint8)
    img_regions = partition_image(lab, n_super_pixels)
    rough_ent = rough_segment(img_regions, ent_bbox_mask, region_merge_thresh)
    # rough_ent = ent_bbox_mask
    # return rough_ent
    ent_segmentation, bgm, fgm = grabcut_from_rough_mask(rough_ent, lab, inv_bg_mask, def_fg, bgm, fgm)
    closing_kernel = np.ones((2, 2), np.uint8)
    ent_segmentation = cv2.morphologyEx(ent_segmentation, cv2.MORPH_OPEN, closing_kernel)
    return ent_segmentation, bgm, fgm


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
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    ent_segmentation, _, _ = segment_entity(anim_frame, scaled_ent_box, bgdModel, fgdModel, None)
    # return ent_segmentation
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
