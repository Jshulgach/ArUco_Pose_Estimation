import cv2

def track_corners(prev_gray, curr_gray, prev_points):
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )

    good_prev = prev_points[status.flatten() == 1]
    good_next = next_points[status.flatten() == 1]

    return good_prev, good_next