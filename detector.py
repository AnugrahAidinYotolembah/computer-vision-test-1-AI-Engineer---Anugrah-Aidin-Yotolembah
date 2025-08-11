# from ultralytics import YOLO
# import numpy as np
# from config import MODEL_PATH, CONF_THRESH, TARGET_CLASS

# class Detector:
#     def __init__(self):
#         try:
#             # Pastikan model yang dimuat adalah model dari Ultralytics (misalnya: yolov8n.pt)
#             self.model = YOLO(MODEL_PATH)
#         except Exception as e:
#             raise RuntimeError(f"Gagal memuat model dari {MODEL_PATH}. "
#                                f"Pastikan model adalah file .pt dari Ultralytics YOLOv8 dan bukan hasil torch.save().\n"
#                                f"Error: {str(e)}")

#     def detect(self, frame):
#         try:
#             results = self.model.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
#         except Exception as e:
#             raise RuntimeError(f"Deteksi gagal dijalankan. Periksa input frame dan model. Error: {str(e)}")

#         detections = []
#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             if cls_id == TARGET_CLASS:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 detections.append([x1, y1, x2, y2, conf])
#         return np.array(detections)

# detector.py
from ultralytics import YOLO
import numpy as np
from config import MODEL_PATH, CONF_THRESH, TARGET_CLASS

class Detector:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model dari {MODEL_PATH}. Error: {str(e)}")

    def detect(self, frame):
        """
        frame: BGR numpy array (OpenCV)
        returns numpy array shape (N,5): [x1,y1,x2,y2,conf]
        """
        try:
            results = self.model.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
        except Exception as e:
            raise RuntimeError(f"Deteksi gagal. Error: {str(e)}")

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        if len(detections) == 0:
            return np.zeros((0,5))
        return np.array(detections)
