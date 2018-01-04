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
tracking_dir = 'tracking'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
segmentation_dir = 'segmentation'
viz_dir = 'viz'

new_dim = 128
owidth = 640
oheight = 480

scale_down = new_dim / owidth
asp_ratio = owidth / oheight


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


def segment_video(video, n_super=1000):
    t_dir = trajectories_dir
    seg_path = os.path.join(t_dir, segmentation_dir)
    frame_arr_data = np.load(os.path.join(t_dir, frame_arr_dir, video.gid() + '.npy'))
    for ent in video.data()['characters'] + video.data()['objects']:
        char_mask = []
        outfile = os.path.join(seg_path, ent.gid() + '_segm.npy')
        entity_rects = np.load(os.path.join(t_dir, tracking_dir, ent.gid() + '.npy'))
        other_ents = [oe for oe in video.data()['characters'] + video.data()['objects'] if oe.gid() != ent.gid()]
        other_rects = [np.load(os.path.join(t_dir, tracking_dir, oe.gid() + '.npy')) for oe in other_ents]
        try:
            for frame_n in range(frame_arr_data.shape[0]):
                scaled_ent_box = scale_box(entity_rects[frame_n])
                scaled_other = [scale_box(oer[frame_n]) for oer in other_rects]
                char_mask.append(segment_entity(frame_arr_data[frame_n], scaled_ent_box, scaled_other, n_super))
        except FileNotFoundError:
            char_mask = np.zeros(frame_arr_data.shape[:3], np.uint8)
        try:
            np.savez_compressed(outfile, np.array(char_mask))
        except FileNotFoundError as e:
            print(e)


def draw_segmentation(segmentation_arr):
    return pil.fromarray(segmentation_arr)


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
    superpixels = slic(img, n_segments, sigma=1, multichannel=True, convert2lab=True, slic_zero=True, enforce_connectivity=False)
    ent_rag = graph.rag_mean_color(img, superpixels)

    regions = graph.merge_hierarchical(superpixels, ent_rag, thresh=10, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=weight_mean_color)
    #     out = color.label2rgb(regions, img, kind='avg')
    #     out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    return regions


def rough_segment(regions, bbox_mask, inclusion_thresh=0.9):
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


def grabcut_from_rough_mask(ent_mask, img):
    mask = np.where(ent_mask == 1, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    ref_mask = np.where((mask == 3), 1, 0).astype('uint8')
    # ref_mask = ref_mask * ent_mask
    # if np.array(combined_other_mask).any():
    #     ref_mask = ref_mask * (combined_other_mask == 0)
    return ref_mask


def segment_entity(frame, ent_rect, n_segments):
    img = deepcopy(frame)
    ent_bbox_mask = create_bbox_segment(ent_rect)
    # for ent_mask in other_bbox_mask[1:]:
    #     combined_other_mask += ent_mask
    img_regions = partition_image(img, n_segments)
    rough_ent = rough_segment(img_regions, ent_bbox_mask)
    ent_segmentation = grabcut_from_rough_mask(rough_ent, img)
    # return ent_segmentation, rough_ent, ent_bbox, mask
    return ent_segmentation


def get_vid_frame_data(vid_gid):
    return np.load(os.path.join(trajectories_dir, frame_arr_dir, vid_gid + '.npy'))


def get_ent_tracking(ent):
    return np.load('./trajectories/tracking/' + ent.gid() + '.npy')


def gen_single_segmentation(video, ent, n_segments=500, frame_n=30):
    test_arr_img = get_vid_frame_data(video.gid())
    anim_frame = test_arr_img[frame_n]
    ent_rects = get_ent_tracking(ent)
    ent_rects = ent_rects[frame_n]
    img = deepcopy(anim_frame)
    scaled_ent_box = scale_box(ent_rects)
    ent_bbox_mask = create_bbox_segment(scaled_ent_box)
    img_regions = partition_image(img, n_segments)
    rough_ent = rough_segment(img_regions, ent_bbox_mask, 0.5)
    ent_segmentation = grabcut_from_rough_mask(rough_ent, img)
    return pil.fromarray(anim_frame * np.tile(np.expand_dims(ent_segmentation, 2), [1, 1, 3]))