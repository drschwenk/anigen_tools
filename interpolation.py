import numpy as np
import os


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
    for i in range(0,anno_frames[0]+1):
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


def generate_interpolation(dataset, eid, interp_dir = 'interpolated_boxes'):
    entity_key_rects = get_entity(dataset, eid).rect()
    entity_rects = np.nan_to_num(interpolate_rects(entity_key_rects, [9,39,69], 75))
    outfile = os.path.join(interp_dir, eid + '.npy')
    np.save(outfile, entity_rects)
    return entity_rects
    

def interpolate_all_video_entites(prod_dataset, video):
    all_eids = [ent.gid() for ent in video.data()['characters'] +  video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    return [generate_interpolation(prod_dataset, eid) for eid in all_eids]
