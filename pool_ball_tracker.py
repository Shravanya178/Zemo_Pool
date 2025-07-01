import cv2
import numpy as np
import math
import os

# ---------------------------
# Configuration and Constants
# ---------------------------

# Video source: use 0 for webcam or filename for video file
VIDEO_SOURCE = "test.mp4"  # e.g., "0" for webcam or path to video

# HoughCircles parameters (tune for your resolution and ball size)
HOUGH_DP = 1.2        # Inverse accumulator resolution
HOUGH_MIN_DIST = 30   # Minimum distance between centers
HOUGH_PARAM1 = 50     # Higher threshold for Canny edge
HOUGH_PARAM2 = 30     # Accumulator threshold (lower = more circles)
HOUGH_MIN_RADIUS = 10 # Min ball radius in pixels (approximate)
HOUGH_MAX_RADIUS = 30 # Max ball radius in pixels (approximate)

# Tracking parameters
MAX_LOST_FRAMES = 5   # how many frames to wait before confirming a ball is lost
DIST_THRESHOLD = 100  # max pixel distance for matching detections to existing track

# White/black detection thresholds in HSV
WHITE_SAT_MAX = 50
WHITE_VAL_MIN = 200
BLACK_VAL_MAX = 50

# Pool ball color-number mapping
POOL_BALL_MAPPING = {
    "Cue": {"number": 0, "color": "White", "type": "Cue"},
    "8-ball": {"number": 8, "color": "Black", "type": "8-ball"},
    # Solid balls (1-7)
    "Yellow": {"number": 1, "color": "Yellow", "type": "Solid"},
    "Blue": {"number": 2, "color": "Blue", "type": "Solid"},
    "Red": {"number": 3, "color": "Red", "type": "Solid"},
    "Purple": {"number": 4, "color": "Purple", "type": "Solid"},
    "Orange": {"number": 5, "color": "Orange", "type": "Solid"},
    "Green": {"number": 6, "color": "Green", "type": "Solid"},
    "Maroon": {"number": 7, "color": "Maroon", "type": "Solid"},
    # Striped balls (9-15)
    "Yellow-Stripe": {"number": 9, "color": "Yellow", "type": "Striped"},
    "Blue-Stripe": {"number": 10, "color": "Blue", "type": "Striped"},
    "Red-Stripe": {"number": 11, "color": "Red", "type": "Striped"},
    "Purple-Stripe": {"number": 12, "color": "Purple", "type": "Striped"},
    "Orange-Stripe": {"number": 13, "color": "Orange", "type": "Striped"},
    "Green-Stripe": {"number": 14, "color": "Green", "type": "Striped"},
    "Maroon-Stripe": {"number": 15, "color": "Maroon", "type": "Striped"}
}

# Global variables for calibration
calibration_data = {
    'table_rect': None,
    'rack_area': None,
    'initial_balls': [],  # List of (center, radius, ball_number, ball_info)
    'calibrated': False,
    'table_corners': None,
    'rack_corners': None
}

# ------------------------
# Helper Classes and Funcs
# ------------------------

class BallTrack:
    """Track state for a single ball."""
    def __init__(self, track_id, center, radius, ball_number, ball_info):
        self.id = track_id
        self.center = center  # (x, y) center of ball
        self.radius = radius
        self.ball_number = ball_number  # Official pool ball number (0-15)
        self.ball_info = ball_info  # Dict with color, type info
        self.lost = 0  # number of consecutive frames not detected
        self.initial_classification = True  # Flag to indicate this was classified in rack
        self.history = [center]  # Track position history for velocity calculation
        
    @property
    def label(self):
        """Generate display label from ball info."""
        if self.ball_number == 0:
            return "Cue Ball"
        elif self.ball_number == 8:
            return "8-Ball"
        else:
            return f"Ball {self.ball_number} ({self.ball_info['color']} {self.ball_info['type']})"
    
    def update_position(self, center, radius):
        """Update ball position and maintain history."""
        self.center = center
        self.radius = radius
        self.history.append(center)
        # Keep only last 10 positions for velocity calculation
        if len(self.history) > 10:
            self.history.pop(0)
        self.lost = 0

def detect_table(frame_hsv):
    """Detect the pool table region by green color thresholding."""
    try:
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        table_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                table_cnt = cnt
                
        if table_cnt is None or max_area < 1000:  # Minimum area threshold
            return None, None
            
        x, y, w, h = cv2.boundingRect(table_cnt)
        return (x, y, w, h), table_cnt
    except Exception as e:
        print(f"Error in detect_table: {e}")
        return None, None

def classify_ball_in_rack(frame, center, radius):
    """
    Classify a ball in the initial rack position.
    This focuses on color detection since numbers face up in the rack.
    """
    try:
        x, y = center
        r = radius
        
        # Extract ball region with some padding
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
        
        if x1 >= x2 or y1 >= y2:
            return None, None
            
        ball_roi = frame[y1:y2, x1:x2]
        
        if ball_roi.size == 0 or ball_roi.shape[0] < 5 or ball_roi.shape[1] < 5:
            return None, None
        
        # Convert to HSV for color analysis
        hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
        
        # Create circular mask
        mask_h, mask_w = ball_roi.shape[:2]
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        center_mask = (mask_w // 2, mask_h // 2)
        mask_radius = min(mask_w, mask_h) // 2 - 2
        if mask_radius > 0:
            cv2.circle(mask, center_mask, mask_radius, 255, -1)
        
        # Analyze colors
        masked_hsv = hsv_roi[mask > 0]
        if masked_hsv.size == 0:
            return None, None
        
        h_vals = masked_hsv[:, 0]
        s_vals = masked_hsv[:, 1]
        v_vals = masked_hsv[:, 2]
        total = float(h_vals.size)
        
        if total == 0:
            return None, None
        
        # Check for white (cue ball) - but cue ball shouldn't be in rack
        white_pixels = np.sum((s_vals < WHITE_SAT_MAX) & (v_vals > WHITE_VAL_MIN))
        black_pixels = np.sum(v_vals < BLACK_VAL_MAX)
        white_frac = white_pixels / total
        black_frac = black_pixels / total
        
        if black_frac > 0.6:  # 8-ball
            return 8, POOL_BALL_MAPPING["8-ball"]
        
        # Determine if striped (has significant white content)
        is_striped = white_frac > 0.25
        
        # Get dominant color from non-white pixels
        color_mask = s_vals > WHITE_SAT_MAX
        if np.sum(color_mask) > 0:
            mean_hue = np.mean(h_vals[color_mask])
        else:
            mean_hue = np.mean(h_vals)
        
        # Map hue to color
        h = mean_hue
        if h < 15 or h > 165:
            color = "Red"
        elif 15 <= h < 30:
            color = "Orange"
        elif 30 <= h < 40:
            color = "Yellow"
        elif 40 <= h < 80:
            color = "Green"
        elif 80 <= h < 130:
            color = "Blue"
        elif 130 <= h < 165:
            color = "Purple"
        else:
            color = "Red"  # Default
        
        # Find corresponding ball number
        ball_key = f"{color}-Stripe" if is_striped else color
        if ball_key in POOL_BALL_MAPPING:
            ball_info = POOL_BALL_MAPPING[ball_key]
            return ball_info["number"], ball_info
        
        return None, None
    except Exception as e:
        print(f"Error in classify_ball_in_rack: {e}")
        return None, None

class CalibrationWindow:
    """Handles the calibration interface for table and rack detection (now with four-corner click)."""
    def __init__(self, frame):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.calibration_step = "table"  # "table" or "rack"
        self.completed = False
        self.window_name = "Pool Table Calibration"
        self.frame_h, self.frame_w = self.frame.shape[:2]
        self.corners = []  # Store clicked corners
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                self.update_display()
                if len(self.corners) == 4:
                    # Store corners and bounding rect
                    xs, ys = zip(*self.corners)
                    rect_x, rect_y = min(xs), min(ys)
                    rect_w, rect_h = max(xs) - rect_x, max(ys) - rect_y
                    if self.calibration_step == "table":
                        calibration_data['table_corners'] = list(self.corners)
                        calibration_data['table_rect'] = (rect_x, rect_y, rect_w, rect_h)
                        self.calibration_step = "rack"
                        self.corners = []
                        print("✓ Table corners selected! Now select rack corners.")
                    elif self.calibration_step == "rack":
                        calibration_data['rack_corners'] = list(self.corners)
                        calibration_data['rack_area'] = (rect_x, rect_y, rect_w, rect_h)
                        self.completed = True
                        print("✓ Rack corners selected! Calibration complete.")
                    self.update_display()
    
    def update_display(self):
        self.display_frame = self.original_frame.copy()
        # Draw existing calibrated areas
        if calibration_data['table_corners']:
            pts = np.array(calibration_data['table_corners'], np.int32).reshape((-1,1,2))
            cv2.polylines(self.display_frame, [pts], isClosed=True, color=(255,0,0), thickness=2)
            cv2.putText(self.display_frame, "TABLE", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        if calibration_data['rack_corners']:
            pts = np.array(calibration_data['rack_corners'], np.int32).reshape((-1,1,2))
            cv2.polylines(self.display_frame, [pts], isClosed=True, color=(0,255,255), thickness=2)
            cv2.putText(self.display_frame, "RACK", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        # Draw current corners
        for i, pt in enumerate(self.corners):
            cv2.circle(self.display_frame, pt, 6, (0,255,0) if self.calibration_step=="table" else (255,255,0), -1)
            cv2.putText(self.display_frame, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if len(self.corners) > 1:
            for i in range(len(self.corners)-1):
                cv2.line(self.display_frame, self.corners[i], self.corners[i+1], (0,255,0) if self.calibration_step=="table" else (255,255,0), 2)
        # Add instruction text with background
        instruction_bg = np.zeros((120, self.display_frame.shape[1], 3), dtype=np.uint8)
        instruction_bg[:] = (50, 50, 50)
        if self.calibration_step == "table":
            text = f"STEP 1: Click 4 corners of the POOL TABLE area ({len(self.corners)}/4)"
            color = (0, 255, 0)
        else:
            text = f"STEP 2: Click 4 corners of the RACK area ({len(self.corners)}/4)"
            color = (255, 255, 0)
        cv2.putText(instruction_bg, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(instruction_bg, "Press 'q' to quit, 'r' to reset, 'ENTER' to skip step", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        status_text = f"Current step: {self.calibration_step.upper()}"
        cv2.putText(instruction_bg, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        self.display_frame = np.vstack([instruction_bg, self.display_frame])
    def run_calibration(self):
        print("\n" + "="*60)
        print("POOL TABLE CALIBRATION")
        print("="*60)
        print("STEP 1: Click 4 corners of the pool table area")
        print("STEP 2: Click 4 corners of the rack area where balls start")
        print("Controls:")
        print("  - Click 4 corners for each area in order")
        print("  - 'q' to quit calibration")
        print("  - 'r' to reset and start over")
        print("  - 'ENTER' to skip current step (use full frame/center)")
        print("="*60)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 700)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()
        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                print("Calibration cancelled by user")
                cv2.destroyWindow(self.window_name)
                return False
            elif key == ord('r'):
                print("Resetting calibration...")
                self.calibration_step = "table"
                self.completed = False
                self.corners = []
                calibration_data['table_rect'] = None
                calibration_data['rack_area'] = None
                calibration_data['table_corners'] = None
                calibration_data['rack_corners'] = None
                self.update_display()
                print("Reset complete. Starting over with table corners...")
            elif key == 13:  # ENTER
                if self.calibration_step == "table":
                    print("Skipping table selection - using full frame")
                    h, w = self.original_frame.shape[:2]
                    calibration_data['table_rect'] = (0, 0, w, h)
                    calibration_data['table_corners'] = [(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
                    self.calibration_step = "rack"
                    self.corners = []
                    self.update_display()
                    print("Now select the rack area...")
                elif self.calibration_step == "rack":
                    print("Skipping rack selection - using center area")
                    if calibration_data['table_rect']:
                        tx, ty, tw, th = calibration_data['table_rect']
                    else:
                        h, w = self.original_frame.shape[:2]
                        tx, ty, tw, th = 0, 0, w, h
                    rack_w, rack_h = tw // 4, th // 4
                    rack_x = tx + tw // 2 - rack_w // 2
                    rack_y = ty + th // 2 - rack_h // 2
                    calibration_data['rack_area'] = (rack_x, rack_y, rack_w, rack_h)
                    calibration_data['rack_corners'] = [
                        (rack_x, rack_y),
                        (rack_x + rack_w, rack_y),
                        (rack_x + rack_w, rack_y + rack_h),
                        (rack_x, rack_y + rack_h)
                    ]
                    self.completed = True
                    self.corners = []
                    self.update_display()
                    print("Using center area as rack region")
            if self.completed:
                print("✓ Calibration completed successfully!")
                cv2.destroyWindow(self.window_name)
                return True
        cv2.destroyWindow(self.window_name)
        return False

def detect_balls_in_rack(frame):
    """Detect and classify all balls in the initial rack formation."""
    if not calibration_data['rack_area']:
        return []
    try:
        rx, ry, rw, rh = calibration_data['rack_area']
        # Validate rack area bounds
        frame_h, frame_w = frame.shape[:2]
        if (rx < 0 or ry < 0 or rx + rw > frame_w or ry + rh > frame_h or rw <= 0 or rh <= 0):
            print("Invalid rack area bounds")
            return []
        rack_roi = frame[ry:ry+rh, rx:rx+rw]
        if rack_roi.size == 0:
            print("Empty rack ROI")
            return []
        # Use HoughCircles to detect balls in rack
        gray = cv2.cvtColor(rack_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST//2,
            param1=HOUGH_PARAM1, param2=max(HOUGH_PARAM2-5, 10),  # Ensure param2 is reasonable
            minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS)
        
        detected_balls = []
        used_numbers = set()
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            print(f"Found {len(circles)} balls in rack area")
            
            for i, (x, y, r) in enumerate(circles):
                # Convert coordinates back to full frame
                full_x, full_y = x + rx, y + ry
                
                # Validate circle bounds
                if (full_x - r < 0 or full_y - r < 0 or 
                    full_x + r >= frame.shape[1] or full_y + r >= frame.shape[0]):
                    continue
                
                # Classify this ball
                ball_number, ball_info = classify_ball_in_rack(frame, (full_x, full_y), r)
                
                # Handle duplicate classifications
                if ball_number and ball_number not in used_numbers:
                    used_numbers.add(ball_number)
                    detected_balls.append((full_x, full_y, r, ball_number, ball_info))
                    print(f"Ball {ball_number} detected at ({full_x}, {full_y})")
                elif ball_number and ball_number in used_numbers:
                    # Find an unused number for similar balls
                    for num in range(1, 16):
                        if num not in used_numbers and num != 8:  # 8-ball is unique
                            used_numbers.add(num)
                            # Create generic ball info
                            ball_type = "Solid" if num <= 7 else "Striped"
                            generic_info = {"number": num, "color": "Unknown", "type": ball_type}
                            detected_balls.append((full_x, full_y, r, num, generic_info))
                            print(f"Ball {num} detected at ({full_x}, {full_y}) [fallback classification]")
                            break
                else:
                    # Couldn't classify - assign generic number
                    for num in range(1, 16):
                        if num not in used_numbers:
                            used_numbers.add(num)
                            ball_type = "Solid" if num <= 7 else "Striped"
                            generic_info = {"number": num, "color": "Unknown", "type": ball_type}
                            detected_balls.append((full_x, full_y, r, num, generic_info))
                            print(f"Ball {num} detected at ({full_x}, {full_y}) [unclassified]")
                            break
        
        calibration_data['initial_balls'] = detected_balls
        calibration_data['calibrated'] = True
        print(f"Initial ball detection complete: {len(detected_balls)} balls identified")
        return detected_balls
    except Exception as e:
        print(f"Error in detect_balls_in_rack: {e}")
        return []

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_near_pocket(center, pockets, threshold=50):
    """Check if a ball is near any pocket."""
    for pocket in pockets:
        if calculate_distance(center, pocket) < threshold:
            return True
    return False

# -------------------------
# Main Processing Pipeline
# -------------------------

def process_video(source=0):
    # Check if video file exists
    if isinstance(source, str) and not source.isdigit():
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found")
            return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("Error: Cannot read from source")
        cap.release()
        return
    print(f"Video loaded: {first_frame.shape[1]}x{first_frame.shape[0]} pixels")
    # === CALIBRATION PHASE ===
    print("Starting calibration...")
    calibrator = CalibrationWindow(first_frame)
    if not calibrator.run_calibration():
        print("Calibration cancelled.")
        cap.release()
        return
    # Detect balls in the initial rack
    initial_balls = detect_balls_in_rack(first_frame)
    if not initial_balls:
        print("No balls detected in rack area. Please recalibrate.")
        cap.release()
        return
    # === TRACKING PHASE ===
    print("Starting tracking phase...")
    next_track_id = 1
    tracks = []
    pocket_events = []
    frame_num = 0
    for ball_data in initial_balls:
        cx, cy, r, ball_number, ball_info = ball_data
        track = BallTrack(next_track_id, (cx, cy), r, ball_number, ball_info)
        tracks.append(track)
        next_track_id += 1
    if calibration_data['table_rect']:
        table_x, table_y, table_w, table_h = calibration_data['table_rect']
    else:
        table_x, table_y, table_w, table_h = 0, 0, first_frame.shape[1], first_frame.shape[0]
    pocket_margin = 30
    pockets = [
        (table_x + pocket_margin, table_y + pocket_margin),
        (table_x + table_w//2, table_y + pocket_margin),
        (table_x + table_w - pocket_margin, table_y + pocket_margin),
        (table_x + pocket_margin, table_y + table_h - pocket_margin),
        (table_x + table_w//2, table_y + table_h - pocket_margin),
        (table_x + table_w - pocket_margin, table_y + table_h - pocket_margin)
    ]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Tracking {len(tracks)} balls...")
    try:
        # Set the main display window to the video resolution
        cv2.namedWindow("Pool Ball Tracker - Calibrated", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pool Ball Tracker - Calibrated", first_frame.shape[1], first_frame.shape[0])
        while True:
            frame_num += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video or read error")
                break
            # For detection, crop to table area, but for display, use full frame
            # Validate table bounds
            if (table_x < 0 or table_y < 0 or table_x + table_w > frame.shape[1] or 
                table_y + table_h > frame.shape[0]):
                print("Invalid table bounds, using full frame")
                table_x, table_y, table_w, table_h = 0, 0, frame.shape[1], frame.shape[0]
            table_roi = frame[table_y:table_y+table_h, table_x:table_x+table_w]
            if table_roi.size == 0:
                print("Empty table ROI, skipping frame")
                continue
            # Detect circles in current frame (on table_roi)
            gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS)
            detections = []
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                for (x, y, r) in circles:
                    cx, cy = int(x + table_x), int(y + table_y)
                    if (cx - r >= 0 and cy - r >= 0 and 
                        cx + r < frame.shape[1] and cy + r < frame.shape[0]):
                        detections.append((cx, cy, int(r)))
            assigned_tracks = set()
            assigned_dets = set()
            for track in tracks:
                best_det = None
                best_dist = float('inf')
                for d_idx, det in enumerate(detections):
                    if d_idx in assigned_dets:
                        continue
                    cx, cy, r = det
                    dist = calculate_distance((cx, cy), track.center)
                    if dist < best_dist and dist < DIST_THRESHOLD:
                        best_dist = dist
                        best_det = d_idx
                if best_det is not None:
                    det = detections[best_det]
                    track.update_position((det[0], det[1]), det[2])
                    assigned_tracks.add(track.id)
                    assigned_dets.add(best_det)
            for track in tracks:
                if track.id not in assigned_tracks:
                    track.lost += 1
            to_remove = []
            for track in tracks:
                if track.lost > 0 and track.lost <= MAX_LOST_FRAMES:
                    if track.lost == 1:
                        if is_near_pocket(track.center, pockets, threshold=60):
                            pocket_events.append((frame_num, track.id, track.ball_number, track.label))
                            print(f"Frame {frame_num}: {track.label} (ID {track.id}) pocketed!")
                            to_remove.append(track)
                elif track.lost > MAX_LOST_FRAMES:
                    print(f"Lost track: {track.label} (ID {track.id})")
                    to_remove.append(track)
            for track in to_remove:
                if track in tracks:
                    tracks.remove(track)
            # === VISUALIZATION ===
            vis = frame.copy()  # Always use full frame for display
            # Draw table boundary
            if calibration_data['table_rect']:
                cv2.rectangle(vis, (table_x, table_y), (table_x + table_w, table_y + table_h), 
                             (255, 0, 0), 2)
                cv2.putText(vis, "Table", (table_x, table_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Draw rack area (for reference)
            if calibration_data['rack_area']:
                rx, ry, rw, rh = calibration_data['rack_area']
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)
                cv2.putText(vis, "Rack Area", (rx, ry - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # Draw pockets
            for i, (px, py) in enumerate(pockets):
                cv2.circle(vis, (px, py), 20, (50, 50, 50), -1)
                cv2.circle(vis, (px, py), 20, (100, 100, 100), 2)
                cv2.putText(vis, f"P{i+1}", (px-10, py+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Draw all detections (for debugging)
            for det in detections:
                cx, cy, cr = det
                cv2.circle(vis, (cx, cy), cr, (128, 128, 128), 1)
            # Draw tracked balls
            for track in tracks:
                cx, cy = track.center
                cr = track.radius
                if track.ball_number == 0:
                    color = (255, 255, 255)
                    text_color = (0, 0, 0)
                elif track.ball_number == 8:
                    color = (0, 0, 0)
                    text_color = (255, 255, 255)
                elif track.ball_number <= 7:
                    color = (0, 255, 0)
                    text_color = (0, 0, 0)
                else:
                    color = (0, 150, 255)
                    text_color = (0, 0, 0)
                if track.lost > 0:
                    color = tuple(int(c * 0.5) for c in color)
                cv2.circle(vis, (int(cx), int(cy)), int(cr), color, 3)
                cv2.putText(vis, str(track.ball_number), (int(cx-10), int(cy+7)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                status = "OK" if track.lost == 0 else f"LOST:{track.lost}"
                label_text = f"ID{track.id}:{status}"
                cv2.putText(vis, label_text, (int(cx-cr), int(cy-cr-15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            info_y = 25
            info_bg_color = (0, 0, 0)
            info_text_color = (255, 255, 255)
            overlay = vis.copy()
            cv2.rectangle(overlay, (5, 5), (300, 150), info_bg_color, -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            cv2.putText(vis, f"Frame: {frame_num}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
            info_y += 25
            cv2.putText(vis, f"Balls on table: {len(tracks)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
            info_y += 25
            cv2.putText(vis, f"Balls pocketed: {len(pocket_events)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
            info_y += 25
            cv2.putText(vis, f"Detections: {len(detections)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
            for i, (fnum, tid, ball_num, lbl) in enumerate(reversed(pocket_events[-3:]), 1):
                info_y += 25
                text = f"Ball {ball_num} pocketed (F{fnum})"
                cv2.putText(vis, text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Pool Ball Tracker - Calibrated", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                print("Recalibrating...")
                calibrator = CalibrationWindow(frame)
                if calibrator.run_calibration():
                    initial_balls = detect_balls_in_rack(frame)
                    if initial_balls:
                        tracks.clear()
                        pocket_events.clear()
                        next_track_id = 1
                        for ball_data in initial_balls:
                            cx, cy, r, ball_number, ball_info = ball_data
                            track = BallTrack(next_track_id, (cx, cy), r, ball_number, ball_info)
                            tracks.append(track)
                            next_track_id += 1
                        print("Recalibration complete!")
                    else:
                        print("Recalibration failed - no balls detected")
            elif key == ord('p'):
                print("Paused - press any key to continue")
                cv2.waitKey(0)
            elif key == ord('s'):
                filename = f"pool_frame_{frame_num:06d}.jpg"
                cv2.imwrite(filename, vis)
                print(f"Frame saved as {filename}")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("\n" + "="*50)
    print("GAME SUMMARY")
    print("="*50)
    print(f"Total frames processed: {frame_num}")
    print(f"Total pocket events: {len(pocket_events)}")
    if pocket_events:
        print("\nPocket Events:")
        for fnum, tid, ball_num, lbl in pocket_events:
            print(f"  Frame {fnum:6d}: Ball {ball_num:2d} pocketed")
    print(f"\nBalls remaining on table: {len(tracks)}")
    if tracks:
        print("Remaining balls:")
        for track in tracks:
            status = "Tracked" if track.lost == 0 else f"Lost ({track.lost} frames)"
            print(f"  Ball {track.ball_number:2d} (ID {track.id}) - {status}")
    print("="*50)

def validate_video_source(source):
    """Validate and return the video source."""
    if isinstance(source, str):
        if source.isdigit():
            return int(source)
        elif os.path.exists(source):
            return source
        else:
            print(f"Video file '{source}' not found.")
            return None
    elif isinstance(source, int):
        return source
    else:
        print("Invalid video source format.")
        return None

def test_calibration_only(source=0):
    """Test just the calibration functionality without full tracking."""
    # Validate video source
    src = validate_video_source(source)
    if src is None:
        return False
    
    # Initialize video capture
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {src}")
        return False
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from video source")
        cap.release()
        return False
    
    print(f"Video loaded successfully: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Test calibration
    print("Testing calibration interface...")
    calibrator = CalibrationWindow(frame)
    
    success = calibrator.run_calibration()
    
    if success:
        print("Calibration test successful!")
        print(f"Table rect: {calibration_data['table_rect']}")
        print(f"Rack area: {calibration_data['rack_area']}")
        
        # Show the calibrated areas on the frame
        vis_frame = frame.copy()
        
        if calibration_data['table_rect']:
            tx, ty, tw, th = calibration_data['table_rect']
            cv2.rectangle(vis_frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 3)
            cv2.putText(vis_frame, "TABLE", (tx + 10, ty + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if calibration_data['rack_area']:
            rx, ry, rw, rh = calibration_data['rack_area']
            cv2.rectangle(vis_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 3)
            cv2.putText(vis_frame, "RACK", (rx + 10, ry + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show result
        cv2.namedWindow("Calibration Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Result", 800, 600)
        cv2.imshow("Calibration Result", vis_frame)
        
        print("Calibration result displayed. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("Calibration test failed or cancelled")
    
    cap.release()
    return success

if __name__ == "__main__":
    print("Pool Ball Tracking System")
    print("=" * 40)
    
    # Ask user what they want to do
    print("Choose an option:")
    print("1. Run full tracking system")
    print("2. Test calibration only")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "2":
        print("Running calibration test...")
        test_calibration_only(VIDEO_SOURCE)
    elif choice == "1":
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Recalibrate")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame")
        print("=" * 40)
        
        # Validate video source
        src = validate_video_source(VIDEO_SOURCE)
        if src is not None:
            try:
                process_video(source=src)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Please check your VIDEO_SOURCE configuration.")
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")
