# import cv2

# # Fungsi untuk menggambar kotak dari tracker
# def draw_boxes(frame, tracks):
#     for track in tracks:
#         x1, y1, x2, y2 = map(int, track.bbox)
#         track_id = track.track_id
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     return frame

# # Fungsi fallback untuk menggambar kotak langsung dari hasil deteksi (tanpa tracking)
# def draw_boxes_from_detections(frame, detections):
#     for det in detections:
#         x1, y1, x2, y2, conf, cls_id = det
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         label = f'ID {int(cls_id)} ({conf:.2f})'
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     return frame

# # Fungsi IoU (tidak diubah)
# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# utils.py
import cv2

def draw_boxes(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def draw_boxes_from_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f'{conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
