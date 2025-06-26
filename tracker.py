pocketed_balls = []
tracked_ids = {}

def track_and_detect_pocketed(balls, pockets, frame):
    global tracked_ids
    new_tracked = {}

    for ball in balls:
        bid = ball['id']
        bx, by = ball['x'], ball['y']
        for px, py in pockets:
            if ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 < 25:
                if bid not in [b[0] for b in pocketed_balls]:
                    pocketed_balls.append((bid, (bx, by), ball.get('label', '')))
        new_tracked[bid] = (bx, by)
    tracked_ids = new_tracked
