import cv2
import numpy as np

ball_id_counter = 0  # Global counter for assigning unique IDs

def detect_balls(frame, table_mask, table_bbox):
    global ball_id_counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=6, maxRadius=15)

    balls = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Fix polygon format for pointPolygonTest
        polygon = np.array([
            [table_bbox[0], table_bbox[1]],
            [table_bbox[2], table_bbox[1]],
            [table_bbox[2], table_bbox[3]],
            [table_bbox[0], table_bbox[3]]
        ], dtype=np.int32).reshape((-1, 1, 2))
        for (x, y, r) in circles:
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                balls.append({'id': ball_id_counter, 'x': x, 'y': y, 'r': r})
                ball_id_counter += 1
    return balls

def classify_balls(frame, balls):
    classified = []
    for ball in balls:
        x, y, r = ball['x'], ball['y'], ball['r']
        roi = frame[max(0, y - r):y + r, max(0, x - r):x + r]
        if roi.size == 0:
            label = ''
        else:
            avg_color = cv2.mean(roi)[:3]
            b, g, r_ = avg_color

            # Cue ball: white with red dots (we just detect as nearly white)
            if r_ > 200 and g > 200 and b > 200:
                label = 'cue'
            elif r_ < 50 and g < 50 and b < 50:
                label = '8'
            elif r_ > 100 and g < 100 and b < 100:
                label = 'solid'
            elif r_ > 100 and g > 100:
                label = 'stripe'
            else:
                label = ''

        classified.append({'id': ball['id'], 'x': x, 'y': y, 'r': r, 'label': label})
    return classified
