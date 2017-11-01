import numpy as np
import os
import cv2
import PIL.Image as pil

trajectories_dir = 'trajectories'
tracking_dir = 'tracking'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
viz_dir = 'viz'

new_dim = 128
o_width = 640
o_height = 480

scale_down = new_dim / o_width
asp_ratio = o_width / o_height


def get_vid_from_eid(eid):
    return '_'.join(eid.split('_')[:7])


def get_entity(dataset, eid):
    vid = get_vid_from_eid(eid)
    video = dataset.get_video(vid)
    entity = [entity for entity in video.data()['characters'] + video.data()['objects'] if entity.gid() == eid]
    if entity:
        return entity[0]
    else:
        return None


def interpolate_rects(anno_rects, anno_frames ,num_frames):
    start_frames_id = [None]*num_frames
    end_frames_id = [None]*num_frames
    for i in range(0, anno_frames[0]+1):
        start_frames_id[i] = 0
        end_frames_id[i] = 1

    for i in range(anno_frames[-1],num_frames):
        start_frames_id[i] = len(anno_frames)-2
        end_frames_id[i] = len(anno_frames)-1

    for j in range(len(anno_frames)-1):
        for i in range(anno_frames[j]+1,anno_frames[j+1]+1):
            start_frames_id[i] = j
            end_frames_id[i] = j+1 

    rects = np.zeros([num_frames, 4])
    for i in range(num_frames):
        rects[i,:] = extrapolate_rect(
            i,
            anno_rects[start_frames_id[i]],
            anno_rects[end_frames_id[i]],
            anno_frames[start_frames_id[i]],
            anno_frames[end_frames_id[i]])[:]

    return rects


def extrapolate_rect(idx, start_rect, end_rect, start_frame, end_frame):
    num_frames = end_frame - start_frame + 1
    rect = [None]*4
    d_start = (idx - start_frame)/(num_frames-1)
    d_end = 1 - d_start
    for j, (c1,c2) in enumerate(zip(start_rect,end_rect)):
        rect[j] = d_end*c1 + d_start*c2
    return rect


def generate_interpolation(dataset, eid):
    try:
        interp_path = os.path.join(trajectories_dir, interp_dir)
        entity_key_rects = get_entity(dataset, eid).rect()
        entity_rects = np.nan_to_num(interpolate_rects(entity_key_rects, [9, 39, 69], 75))
        outfile = os.path.join(interp_path, eid + '.npy')
        np.save(outfile, entity_rects)
    except FileNotFoundError:
        print(eid)
    return entity_rects
    

def interpolate_all_video_entites(prod_dataset, video):
    all_eids = [ent.gid() for ent in video.data()['characters'] + video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    return [generate_interpolation(prod_dataset, eid) for eid in all_eids]


def draw_all_bboxes(frame_arr_square, raw_bboxes, entity_type='character'):
    color_assignments = {
        'character': (0, 255, 255),
        'object': (0, 255, 0),
    }
    frame_arr = cv2.resize(frame_arr_square, None, fx = asp_ratio, fy=1)
    bboxes = [bb.reshape(2, 2) for bb in raw_bboxes]
    for bb in bboxes:
        bb[:, 0] = bb[:, 0] * scale_down * asp_ratio
        bb[:, 1] = bb[:, 1] * scale_down * asp_ratio
    bboxes = [bb.astype(int) for bb in bboxes]
    _ = [cv2.rectangle(frame_arr, tuple(bb[0]), tuple(bb[1]), color_assignments[entity_type] , thickness=1) for bb in bboxes]
    return pil.fromarray(frame_arr)


def draw_all_video_entites(prod_dataset, video):
    all_eids = [ent.gid() for ent in video.data()['characters'] + video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    return [draw_video_interps(prod_dataset, eid) for eid in all_eids]


def draw_video_interps(video, retrieved=True):
    if retrieved:
        t_dir = './retrieved/' + trajectories_dir
    else:
        t_dir = trajectories_dir
    outfile = os.path.join(t_dir, viz_dir, video.gid() + '_interp.gif')
    all_eids = [ent.gid() for ent in video.data()['characters'] + video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    entity_interps = [np.load(os.path.join(t_dir,  interp_dir, eid + '.npy')) for eid in all_eids]
    frame_arr_data = np.load(os.path.join(t_dir,  frame_arr_dir, video.gid() + '.npy'))
    interp_img_seq = [draw_all_bboxes(frame_arr_data[frame_n], [entity_rect[frame_n] for entity_rect in entity_interps],
                                      'object') for frame_n in range(frame_arr_data.shape[0])]
    interp_img_seq[0].save(outfile, save_all=True, optimize=True, duration=42, append_images=interp_img_seq[1:])
    return
