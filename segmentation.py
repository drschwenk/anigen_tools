import numpy as np
import cv2


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
    mask, bgd_model, fgd_model = cv2.grabCut(img,mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    return img


def segment_entity_rect(img, rect):
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
    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    final_mask = mask2[:, :, np.newaxis] * cleanup_mask[:, :, np.newaxis]
    img = img * final_mask
    return img, final_mask
