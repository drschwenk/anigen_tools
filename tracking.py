from skimage.feature import match_template
import os
import numpy as np
import PIL.Image as pil
import cv2
import copy

trajectories_dir = 'trajectories'
tracking_dir = 'tracking'
interp_dir = 'interpolation'
frame_arr_dir = 'frame_arr_data'
viz_dir = 'viz'

new_dim = 128
owidth = 640
oheight = 480

scale_down = new_dim / owidth
asp_ratio = owidth / oheight


def track(frames,boxes,anno_frame_ids,search_region_scale_factor,img_size):
    boxes = boxes.astype(np.float32)

    # Template box
    tx1, ty1, tx2, ty2 = boxes[anno_frame_ids[0],:].astype(np.int32)
    template = frames[anno_frame_ids[0], ty1:ty2, tx1:tx2, :]
    for i in range(anno_frame_ids[0]):
        boxes[i, :] = track_inner(
            frames[i],
            boxes[i, :],
            template,
            search_region_scale_factor,
            img_size)

    for j in range(len(anno_frame_ids)-1):
        template_choices = [None]*2
        tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[j],:].astype(np.int32)
        template_choices[0] = frames[anno_frame_ids[j],ty1:ty2,tx1:tx2,:]
        tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[j+1],:].astype(np.int32)
        template_choices[1] = frames[anno_frame_ids[j+1],ty1:ty2,tx1:tx2,:]
        for i in range(anno_frame_ids[j]+1,anno_frame_ids[j+1]):
            if i-anno_frame_ids[j] < anno_frame_ids[j+1]-i:
                template = template_choices[0]
            else:
                template = template_choices[1]

            boxes[i,:] = track_inner(
                frames[i],
                boxes[i,:],
                template,
                search_region_scale_factor,
                img_size)

    tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[-1],:].astype(np.int32)
    template = frames[anno_frame_ids[-1],ty1:ty2,tx1:tx2,:]
    for i in range(anno_frame_ids[-1],frames.shape[0]):
        boxes[i,:] = track_inner(
            frames[i],
            boxes[i,:],
            template,
            search_region_scale_factor,
            img_size)

    return boxes


def track_inner(frame,box,template,search_region_scale_factor,img_size):
    im_h,im_w = img_size
    x1,y1,x2,y2 = box
    w = search_region_scale_factor*(x2-x1)
    h = search_region_scale_factor*(y2-y1)
    x1_ = int(max(0,0.5*(x1+x2-w)))
    x2_ = int(min(im_w,0.5*(x1+x2+w)))
    y1_ = int(max(0,0.5*(y1+y2-h)))
    y2_ = int(min(im_h,0.5*(y1+y2+h)))
    frame = frame[y1_:y2_,x1_:x2_,:]
    if any([v==0 for v in template.shape]) or any([v==0 for v in frame.shape]):
        return box

    # print(template.shape, frame.shape)
    try:
        result = match_template(frame,template,pad_input=True)
    except ValueError as e:
        return box
    h, w, _ = result.shape

    xa, ya = np.meshgrid(range(w), range(h))
    xa = (xa - w/2) * 2 / w
    ya = (ya - h/2) * 2 / h
    pos_term = np.exp(-(xa**2 + ya**2))
    pos_term = np.tile(np.expand_dims(pos_term, 2), [1, 1, 3])
    result += pos_term

    ij = np.unravel_index(np.argmax(result), result.shape)
    cy,cx = ij[0:2]
    del_x = cx + x1_ - 0.5*(x1+x2)
    del_y = cy + y1_ - 0.5*(y1+y2)
    x1,x2 = [x+del_x for x in [x1,x2]]
    y1,y2 = [y+del_y for y in [y1,y2]]
    box = [x1,y1,x2,y2]

    return box


def scale_boxes(boxes,in_frame_size,out_frame_size):
    hi,wi = in_frame_size
    ho,wo = out_frame_size
    fh = ho/float(hi)
    fw = wo/float(wi)
    boxes = np.array(boxes).astype(np.float32)
    for i in [0,2]:
        boxes[:,i] *= fw
        boxes[:,i+1] *= fh

    boxes = boxes.astype(np.int32)

    return boxes


def track_all_video_entites(video):
    frame_arr_data = np.load(os.path.join(trajectories_dir, frame_arr_dir, video.gid() + '.npy'))
    all_eids = [ent.gid() for ent in video.data()['characters'] + video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    return [generate_tracking(eid, frame_arr_data) for eid in all_eids]


def track_rects(entity_key_rects, frame_arr_data):
    boxes = scale_boxes(entity_key_rects,(480,640),(128,128))
    boxes = track(frame_arr_data,boxes,[9,39,69],2.0,(128,128))
    rescaled_boxes = scale_boxes(boxes,(128,128), (480,640))
    return rescaled_boxes


def generate_tracking(eid, frame_arr_data):
    boxes_npy = os.path.join(trajectories_dir, interp_dir, eid + '.npy')
    boxes = np.load(boxes_npy)
    entity_rects = np.nan_to_num(track_rects(boxes, frame_arr_data))
    outfile = os.path.join(trajectories_dir, tracking_dir, eid + '.npy')
    np.save(outfile, entity_rects)
    return entity_rects


def draw_all_bboxes(frame_arr_square, raw_bboxes, entity_type = 'character'):
    color_assignments = {
        'character': (0, 255, 255),
        'object': (0, 255, 0),
    }
    frame_arr = cv2.resize(frame_arr_square, None, fx=asp_ratio, fy=1)
    bboxes = copy.deepcopy([bb.reshape(2, 2) for bb in raw_bboxes])
    for bb in bboxes:
        bb[:, 0] = bb[:, 0] * scale_down * asp_ratio
        bb[:, 1] = bb[:, 1] * scale_down * asp_ratio
    bboxes = [bb.astype(int) for bb in bboxes]
    _ = [cv2.rectangle(frame_arr, tuple(bb[0]), tuple(bb[1]), color_assignments[entity_type] , thickness=1) for bb in bboxes]
    return pil.fromarray(frame_arr)


# def draw_video_tracking(video):
#     outfile = os.path.join(trajectories_dir, viz_dir, video.gid() + '_tracking.gif')
#     frame_arr_data = np.load(os.path.join(trajectories_dir,  frame_arr_dir, video.gid() + '.npy'))
#     try:
#         entity_interps = track_all_video_entites(video)
#         interp_img_seq = [draw_all_bboxes(frame_arr_data[frame_n], [entity_rect[frame_n] for entity_rect in entity_interps],
#             'object') for frame_n in range(frame_arr_data.shape[0])]
#         interp_img_seq[0].save(outfile, save_all=True, optimize=True, duration=42, append_images=interp_img_seq[1:])
#         return interp_img_seq
#     except ValueError as e:
#         print(video.gid(), '\n', e)


def draw_video_tracking(video, retrieved=True):
    tracking_dir = 'tracking_stabilized'
    if retrieved:
        t_dir = './retrieved/' + trajectories_dir
    else:
        t_dir = trajectories_dir
    outfile = os.path.join('viz', video.gid() + '_tracking.gif')
    all_eids = [ent.gid() for ent in video.data()['characters'] + video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    entity_interps = [np.load(os.path.join(t_dir,  tracking_dir, eid + '.npy')) for eid in all_eids]
    frame_arr_data = np.load(os.path.join(t_dir,  frame_arr_dir, video.gid() + '.npy'))
    interp_img_seq = [draw_all_bboxes(frame_arr_data[frame_n], [entity_rect[frame_n] for entity_rect in entity_interps],
                                      'object') for frame_n in range(frame_arr_data.shape[0])]
    interp_img_seq[0].save(outfile, save_all=True, optimize=True, duration=42, append_images=interp_img_seq[1:])
    return
