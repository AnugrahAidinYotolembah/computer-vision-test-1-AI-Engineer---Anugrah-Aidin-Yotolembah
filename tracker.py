# import numpy as np
# from collections import deque
# from utils import iou

# class Track:
#     def __init__(self, bbox, track_id):
#         self.bbox = bbox
#         self.track_id = track_id
#         self.hits = 0
#         self.miss = 0

#     def update(self, bbox):
#         self.bbox = bbox
#         self.hits += 1
#         self.miss = 0

# class Tracker:
#     def __init__(self, iou_threshold=0.3, max_lost=5):
#         self.tracks = []
#         self.next_id = 0
#         self.iou_threshold = iou_threshold
#         self.max_lost = max_lost

#     def update(self, detections, frame_info=None):
#         updated_tracks = []

#         for det in detections:
#             matched = False
#             for track in self.tracks:
#                 if iou(det[:4], track.bbox) > self.iou_threshold:
#                     track.update(det[:4])
#                     updated_tracks.append(track)
#                     matched = True
#                     break

#             if not matched:
#                 new_track = Track(det[:4], self.next_id)
#                 self.next_id += 1
#                 updated_tracks.append(new_track)

#         for track in self.tracks:
#             if track not in updated_tracks:
#                 track.miss += 1
#                 if track.miss < self.max_lost:
#                     updated_tracks.append(track)

#         self.tracks = updated_tracks
#         return self.tracks

# tracker.py
import numpy as np
from collections import deque
from utils import iou

try:
    # try to use scipy for optimal assignment
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = np.array(bbox, dtype=float)  # [x1,y1,x2,y2]
        self.track_id = int(track_id)
        self.hits = 1
        self.miss = 0
        self.max_history = 30
        self.history = deque(maxlen=self.max_history)

    def update(self, bbox):
        self.bbox = np.array(bbox, dtype=float)
        self.hits += 1
        self.miss = 0
        self.history.append(self.bbox.copy())

    def mark_missed(self):
        self.miss += 1

class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def _associate(self, detections):
        """
        Return matches list of (det_idx, track_idx), unmatched_dets, unmatched_tracks
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=float)
        for d, det in enumerate(detections):
            for t, tr in enumerate(self.tracks):
                iou_matrix[d, t] = iou(det[:4], tr.bbox)

        # convert to cost (maximize IoU -> minimize -IoU)
        cost = -iou_matrix

        matches = []
        if SCIPY_AVAILABLE:
            det_idx, tr_idx = linear_sum_assignment(cost)
            for d, t in zip(det_idx, tr_idx):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matches.append((d, t))
        else:
            # greedy fallback
            used_d = set()
            used_t = set()
            sorted_pairs = []
            for d in range(iou_matrix.shape[0]):
                for t in range(iou_matrix.shape[1]):
                    sorted_pairs.append((iou_matrix[d,t], d, t))
            sorted_pairs.sort(reverse=True, key=lambda x: x[0])  # highest IoU first
            for val, d, t in sorted_pairs:
                if d in used_d or t in used_t:
                    continue
                if val >= self.iou_threshold:
                    used_d.add(d); used_t.add(t)
                    matches.append((d,t))

        matched_dets = [m[0] for m in matches]
        matched_trs = [m[1] for m in matches]

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trs = [t for t in range(len(self.tracks)) if t not in matched_trs]

        return matches, unmatched_dets, unmatched_trs

    def update(self, detections, frame_info=None):
        """
        detections: numpy array (N,5) [x1,y1,x2,y2,conf]
        """
        detections = list(detections) if len(detections) else []
        matches, unmatched_dets, unmatched_trs = self._associate(detections)

        updated = []

        # update matched tracks
        for d_idx, t_idx in matches:
            bbox = detections[d_idx][:4]
            tr = self.tracks[t_idx]
            tr.update(bbox)
            updated.append(tr)

        # create new tracks for unmatched detections
        for d in unmatched_dets:
            bbox = detections[d][:4]
            new_tr = Track(bbox, self.next_id)
            self.next_id += 1
            updated.append(new_tr)

        # increase miss count for unmatched tracks and keep if not expired
        for t in unmatched_trs:
            tr = self.tracks[t]
            tr.mark_missed()
            if tr.miss < self.max_lost:
                updated.append(tr)
            # else drop track

        self.tracks = updated
        return self.tracks
