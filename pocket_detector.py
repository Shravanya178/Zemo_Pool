import cv2

def detect_pockets(frame, mask, table_bbox):
    height, width = frame.shape[:2]
    pocket_radius = 20

    pockets = [
        (table_bbox[0], table_bbox[1]),
        (table_bbox[2], table_bbox[1]),
        (table_bbox[0], table_bbox[3]),
        (table_bbox[2], table_bbox[3]),
        ((table_bbox[0] + table_bbox[2]) // 2, table_bbox[1]),
        ((table_bbox[0] + table_bbox[2]) // 2, table_bbox[3]),
    ]
    return pockets
