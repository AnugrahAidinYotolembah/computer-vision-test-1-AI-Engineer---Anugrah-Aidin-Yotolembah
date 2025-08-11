# # # RTSP_URL = "rtsp://localhost:8554/live/mystream"
# # RTSP_URL = "rtsp://rtspstream:Y9RQEg9q8hn_esZjbT_Xl@zephyr.rtsp.stream/people"


# MODEL_PATH = "yolov8n.pt"
# CONF_THRESH = 0.3
# TARGET_CLASS = 0  # People class (COCO)

# config.py

# Model & detection
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.3
TARGET_CLASS = 0  # People class (COCO)

# Runtime / performance
FRAME_WIDTH = 960        # ukuran default untuk upload video (resized)
FRAME_HEIGHT = 540
RTSP_FRAME_WIDTH = 640   # ukuran default untuk RTSP processing
RTSP_FRAME_HEIGHT = 360

# Frame skipping: deteksi hanya tiap N frame
RTSP_DETECT_EVERY = 3
UPLOAD_DETECT_EVERY = 8

# RTSP reconnect/backoff
RTSP_MAX_RETRIES = 10
RTSP_RETRY_INTERVAL = 2.0  # detik, base interval (bertambah tiap retry)

# Logging files
LOG_RTSP = "log_rtsp.csv"
LOG_UPLOAD = "log_upload.csv"
