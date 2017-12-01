import numpy as np
import cv2
import os
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


def segment_entity(img, rect, bg_mask):
    mask = bg_mask
    img = np.array(img)[:, :, ::]
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask, bgd_model, fgd_model = cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    return img


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
    bb = bbox.reshape(2, 2)
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


def segment_from_quant_img(video, frame_arr_data):
    t_dir = trajectories_dir
    seg_path = os.path.join(t_dir, segmentation_dir)
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


def draw_segmentation(segmentation_arr):
    return pil.fromarray(segmentation_arr)


def degrade_colorspace(image):
    return


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
            outfile = os.path.join(seg_path, ent.gid() + '_segm.gif').replace(' ', '_')
            char_mask = np.load(os.path.join(t_dir,  segmentation_dir, ent.gid() + '_segm.npy.npz'))
            char_mask = np.expand_dims(char_mask['arr_0'], 3)
            segm_arr = frame_arr_data * char_mask
            segmentation_frames = [draw_segmentation(segm_arr[frame_n]) for frame_n in
                                    range(frame_arr_data.shape[0])]
            segmentation_frames[0].save(outfile, save_all=True, optimize=True, duration=42,
                                   append_images=segmentation_frames[1:])
    # except:
    #     print(video.gid())