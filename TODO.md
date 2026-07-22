- the real task of the ML model is to find out if drone is strictly in the middle of the screen and nothing else, so it needs to find only one object in the frame + it cares only about aligning of center of the drone and center of the screen - so the idea is to change loss function by adding more weight to coordinate precision and less weight to bounding boxes

Postprocessing flow:
1. Select seed box. From the pre-NMS boxes, keep those with conf > thresh_conf; pick the one closest to screen center. (If None → close_enough = False, exit.)
2. Gather cluster. Collect all boxes with IOU > thresh_iou against the seed box.
3. WBF the cluster → single measured box.
4. Kalman predict → predicted box.
5. Gate. IOU(measured, predicted) > thresh_frame_iou? 
    Yes → miss_counter = 0
    No → miss_counter += 1 
        # add measured box to list of previous boxes
        if miss_counter > M 
            re-seed filter wit measured box
            # take all last consecutive boxes with IOU(measured, prev) > thresh_frame_iou
            # re-seed filter with selected boxes 
    close_enough = False, exit.
6. Kalman update → smoothed center → offset from center (to aiming) and its distance dist.
7. Centering gate on dist — hysteresis (D_low/D_high) + debounce (N) → close_enough.