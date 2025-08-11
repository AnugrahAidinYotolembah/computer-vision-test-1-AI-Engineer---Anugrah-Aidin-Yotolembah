# import cv2
# import time

# class RTSPStream:
#     def __init__(self, url):
#         self.url = url
#         self.cap = None
#         self.connect()

#     def connect(self):
#         if self.cap:
#             self.cap.release()
#         self.cap = cv2.VideoCapture(self.url)
#         retries = 0
#         while not self.cap.isOpened() and retries < 5:
#             time.sleep(2)
#             self.cap = cv2.VideoCapture(self.url)
#             retries += 1

#     def read(self):
#         if not self.cap.isOpened():
#             self.connect()
#         ret, frame = self.cap.read()
#         if not ret:
#             self.connect()
#             return None
#         return frame

# # rtsp_handler.py
# import cv2
# import threading
# import time
# import logging

# from config import RTSP_MAX_RETRIES, RTSP_RETRY_INTERVAL

# logging.getLogger().setLevel(logging.INFO)

# class RTSPStream:
#     """
#     Background RTSP reader with auto-reconnect and non-blocking read().
#     - connect() mencoba membuka VideoCapture dengan retry/backoff.
#     - background thread membaca frame terakhir terus menerus.
#     - read() mengembalikan frame terbaru (None jika tidak tersedia).
#     """
#     def __init__(self, url, name="rtsp"):
#         self.url = url
#         self.name = name
#         self.cap = None
#         self.frame = None
#         self._running = False
#         self._thread = None
#         self._lock = threading.Lock()
#         self.connect()

#     def connect(self):
#         # Try to open capture with retries and exponential backoff
#         retries = 0
#         interval = RTSP_RETRY_INTERVAL
#         if self.cap:
#             try:
#                 self.cap.release()
#             except:
#                 pass
#         self.cap = cv2.VideoCapture(self.url)
#         while not self.cap.isOpened() and retries < RTSP_MAX_RETRIES:
#             logging.info(f"[RTSPStream] connect(): failed to open {self.url}, retry {retries+1}/{RTSP_MAX_RETRIES}")
#             time.sleep(interval)
#             retries += 1
#             interval *= 1.5
#             self.cap = cv2.VideoCapture(self.url)
#         if self.cap.isOpened():
#             logging.info(f"[RTSPStream] connect(): opened {self.url}")
#         else:
#             logging.warning(f"[RTSPStream] connect(): unable to open {self.url} after {retries} retries")

#         # start reader thread if not running
#         if not self._running:
#             self._running = True
#             self._thread = threading.Thread(target=self._reader, daemon=True)
#             self._thread.start()

#     def _reader(self):
#         # continuously read frames; try reconnect if read fails
#         while self._running:
#             if not self.cap or not self.cap.isOpened():
#                 # try reconnect
#                 logging.info(f"[RTSPStream] reader: trying to reconnect {self.url}")
#                 self.connect()
#                 time.sleep(1.0)
#                 continue

#             ret, frame = self.cap.read()
#             if not ret or frame is None:
#                 logging.warning(f"[RTSPStream] reader: read failed, reconnecting...")
#                 try:
#                     self.cap.release()
#                 except:
#                     pass
#                 self.cap = None
#                 # short sleep then reconnect in connect()
#                 time.sleep(1.0)
#                 continue

#             # store latest frame (thread-safe)
#             with self._lock:
#                 self.frame = frame

#             # small sleep to avoid tight loop (capture has builtin pacing)
#             time.sleep(0.01)

#     def read(self):
#         """
#         Return latest frame (copy) or None.
#         Non-blocking.
#         """
#         with self._lock:
#             if self.frame is None:
#                 return None
#             # return a copy to avoid race conditions
#             return self.frame.copy()

#     def stop(self):
#         self._running = False
#         if self._thread and self._thread.is_alive():
#             self._thread.join(timeout=1.0)
#         if self.cap:
#             try:
#                 self.cap.release()
#             except:
#                 pass
#         logging.info(f"[RTSPStream] stopped {self.url}")

# rtsp_handler.py
import cv2
import threading
import time
import logging

from config import RTSP_MAX_RETRIES, RTSP_RETRY_INTERVAL

logging.getLogger().setLevel(logging.INFO)

class RTSPStream:
    """
    Background RTSP reader with auto-reconnect and non-blocking read().
    - connect() tries to open VideoCapture with retry/backoff.
    - background thread reads frames continuously.
    - read() returns latest frame (copy) or None.
    """
    def __init__(self, url, name="rtsp"):
        self.url = url
        self.name = name
        self.cap = None
        self.frame = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self.connect()

    def connect(self):
        with self._connect_lock:
            retries = 0
            interval = RTSP_RETRY_INTERVAL
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.url)
            while not self.cap.isOpened() and retries < RTSP_MAX_RETRIES:
                logging.info(f"[RTSPStream] connect(): failed to open {self.url}, retry {retries+1}/{RTSP_MAX_RETRIES}")
                time.sleep(interval)
                retries += 1
                interval *= 1.5
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                logging.info(f"[RTSPStream] connect(): opened {self.url}")
            else:
                logging.warning(f"[RTSPStream] connect(): unable to open {self.url} after {retries} retries")

            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._reader, daemon=True)
                self._thread.start()

    def _reader(self):
        while self._running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logging.info(f"[RTSPStream] reader: cap not opened, trying to connect {self.url}")
                    self.connect()
                    time.sleep(1.0)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning(f"[RTSPStream] reader: read failed, attempting reconnect...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(1.0)
                    continue

                with self._lock:
                    # store latest frame (copy to be safe)
                    self.frame = frame.copy()

                # small sleep to avoid tight loop; capture pacing persists
                time.sleep(0.005)
            except Exception as e:
                logging.exception("[RTSPStream] reader exception: %s", e)
                time.sleep(0.5)

    def read(self):
        """Return latest frame copy or None (non-blocking)."""
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        """Stop reader thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        logging.info(f"[RTSPStream] stopped {self.url}")
