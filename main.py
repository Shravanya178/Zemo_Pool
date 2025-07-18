import argparse
import cv2
from table_detector import detect_table
from ball_detector import detect_balls, classify_balls
from pocket_detector import detect_pockets

# Persistent state
persistent_labels = {}  # ball_id -> label
pocketed_ids = set()    # already pocketed ball IDs
pocketed_log = []       # list of (id, label)

def is_ball_pocketed(ball, pockets, threshold=25):
    for (px, py) in pockets:
        dist = ((ball['x'] - px) ** 2 + (ball['y'] - py) ** 2) ** 0.5
        if dist < threshold:
            return True
    return False

def draw_annotations(frame, balls, pockets):
    # Draw pockets
    for (px, py) in pockets:
        cv2.circle(frame, (int(px), int(py)), 18, (0, 0, 0), 3)

    # Draw balls with labels
    for ball in balls:
        x, y, r = int(ball['x']), int(ball['y']), int(ball['r'])
        label = persistent_labels.get(ball['id'], '')

        if label == 'cue':
            color = (255, 255, 255)
            text_color = (0, 0, 0)
        elif label == '8':
            color = (0, 0, 0)
            text_color = (255, 255, 255)
        elif label == 'stripe':
            color = (0, 255, 255)
            text_color = (0, 0, 0)
        elif label == 'solid':
            color = (255, 0, 0)
            text_color = (255, 255, 255)
        else:
            color = (0, 255, 0)
            text_color = (0, 0, 0)

        cv2.circle(frame, (x, y), r, color, 2)
        cv2.putText(frame, label, (x - r, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Draw pocketed log
    y0 = 30
    for i, (pid, label) in enumerate(pocketed_log[-10:]):  # last 10 pocketed
        cv2.putText(frame, f"Pocketed: {label}", (10, y0 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="8 Ball Pool Video Analytics")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("output_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Read first frame for table and pockets
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    table_mask, table_bbox = detect_table(first_frame)
    pockets = detect_pockets(first_frame, table_mask, table_bbox)

    # Detect and classify all balls in the first frame
    initial_balls = detect_balls(first_frame, table_mask, table_bbox)
    classified = classify_balls(first_frame, initial_balls)

    # Store persistent labels
    for ball in classified:
        persistent_labels[ball['id']] = ball['label']

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        balls = detect_balls(frame, table_mask, table_bbox)
        for ball in balls:
            ball_id = ball['id']
            label = persistent_labels.get(ball_id, '')
            ball['label'] = label

            # Check if pocketed
            if ball_id not in pocketed_ids and is_ball_pocketed(ball, pockets):
                pocketed_ids.add(ball_id)
                pocketed_log.append((ball_id, label))
                print(f"Ball pocketed: ID={ball_id}, Label={label}")

        annotated = draw_annotations(frame.copy(), balls, pockets)
        out.write(annotated)
        cv2.imshow("8 Ball Pool Analytics", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
