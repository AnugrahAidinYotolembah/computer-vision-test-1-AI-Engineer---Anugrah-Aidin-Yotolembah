# dashboard.py (final — fixed types, per-stream trackers, safe threads)
import streamlit as st
import tempfile
import cv2
import os
import time
import csv
import threading
import queue
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from detector import Detector
from tracker import Tracker
from rtsp_handler import RTSPStream
from utils import draw_boxes
from config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT,
    RTSP_DETECT_EVERY, UPLOAD_DETECT_EVERY,
    LOG_RTSP, LOG_UPLOAD
)

# extra log for webcam
LOG_WEBRTC = "log_webrtc.csv"

# -------------------- helpers: logging --------------------
def init_log_file(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "message"])

def write_log_csv(path, level, message):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, message])

# -------------------- initial setup --------------------
st.set_page_config(page_title="CCTV AI Multi-Mode", layout="wide")
st.title("📹 CCTV AI Pipeline :  RTSP Real-time Detection & Tracking.")
st.markdown(
    "<p style='font-size:12px; color:gray; margin-top:-10px;'>by Anugrah Aidin Yotolembah</p>",
    unsafe_allow_html=True
)


# prepare shared logs
init_log_file(LOG_UPLOAD)
init_log_file(LOG_WEBRTC)

# ---------------- session state defaults --------------------
if "rtsp_thread_1" not in st.session_state:
    st.session_state.rtsp_thread_1 = None
if "rtsp_thread_2" not in st.session_state:
    st.session_state.rtsp_thread_2 = None

if "rtsp_running_1" not in st.session_state:
    st.session_state.rtsp_running_1 = False
if "rtsp_running_2" not in st.session_state:
    st.session_state.rtsp_running_2 = False

# create stop events (threading.Event) persisted in session_state
if "rtsp_stop_event_1" not in st.session_state:
    st.session_state.rtsp_stop_event_1 = threading.Event()
if "rtsp_stop_event_2" not in st.session_state:
    st.session_state.rtsp_stop_event_2 = threading.Event()

# frame queues (one-slot) per cam -> worker puts latest annotated BGR frame, main thread reads
if "frame_queue_1" not in st.session_state:
    st.session_state.frame_queue_1 = queue.Queue(maxsize=1)
if "frame_queue_2" not in st.session_state:
    st.session_state.frame_queue_2 = queue.Queue(maxsize=1)

# -------------------- utility: normalize detections --------------------
def normalize_detections(dets):
    """
    Ensure detections are a numpy array with shape (N,5) where each row is [x1,y1,x2,y2,conf].
    If dets is None or empty, return np.zeros((0,5), dtype=float).
    """
    if dets is None:
        return np.zeros((0,5), dtype=float)
    if isinstance(dets, np.ndarray):
        if dets.size == 0:
            return np.zeros((0,5), dtype=float)
        return dets
    # if list of boxes -> try convert
    try:
        arr = np.asarray(dets)
        if arr.ndim == 1 and arr.size == 0:
            return np.zeros((0,5), dtype=float)
        # If shape (N,5) or (N,4) handle
        if arr.ndim == 2 and arr.shape[1] >= 5:
            return arr[:, :5].astype(float)
        elif arr.ndim == 2 and arr.shape[1] == 4:
            # no conf column: append conf=1.0
            confs = np.ones((arr.shape[0],1), dtype=float)
            return np.hstack([arr.astype(float), confs])
        else:
            # fallback: empty
            return np.zeros((0,5), dtype=float)
    except Exception:
        return np.zeros((0,5), dtype=float)

# -------------------- RTSP worker (no Streamlit calls inside) --------------------
def rtsp_worker(rtsp_url, cam_id, log_path, width, height, detect_every, stop_event, frame_queue):
    """
    Background worker for RTSP camera (runs in separate thread).
    Does NOT call any st.* functions — only writes logs and puts frames into frame_queue.
    """
    # instantiate per-thread detector & tracker to avoid shared-model thread-safety issues
    detector_local = Detector()
    tracker_local = Tracker()

    # create RTSP reader (uses your rtsp_handler)
    try:
        rtsp = RTSPStream(rtsp_url)
    except Exception as e:
        write_log_csv(log_path, "ERROR", f"Failed to create RTSPStream: {e}")
        return

    frame_count = 0
    last_detections = np.zeros((0,5), dtype=float)

    # init log file and write start info
    init_log_file(log_path)
    write_log_csv(log_path, "INFO", f"RTSP worker started for cam{cam_id}: {rtsp_url}")

    while not stop_event.is_set():
        frame = rtsp.read()
        if frame is None:
            # wait briefly then retry (handles short disconnects)
            time.sleep(0.1)
            continue

        frame_count += 1
        # resize for processing (keep BGR)
        try:
            proc = cv2.resize(frame, (width, height))
        except Exception:
            proc = frame

        # detection with frame skipping
        if frame_count % detect_every == 0:
            try:
                raw = detector_local.detect(proc)
            except Exception as e:
                raw = None
                write_log_csv(log_path, "ERROR", f"Detect error: {str(e)}")
            dets = normalize_detections(raw)
            last_detections = dets
            write_log_csv(log_path, "DETECTION", f"Frame {frame_count} - {len(dets)} objek")
        else:
            dets = last_detections

        # tracker update (use per-thread tracker_local)
        try:
            tracks = tracker_local.update(dets, {"img_shape": proc.shape, "img_size": proc.shape[:2]})
        except Exception as e:
            tracks = []
            write_log_csv(log_path, "ERROR", f"Tracker update error: {e}")

        annotated = draw_boxes(proc, tracks)

        # put latest annotated frame into queue (replace old if full)
        try:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(annotated)
        except Exception:
            # ignore queue errors
            pass

        # small throttle
        time.sleep(0.01)

    # cleanup
    try:
        rtsp.stop()
    except Exception:
        pass
    write_log_csv(log_path, "INFO", f"RTSP worker stopped for cam{cam_id}")

# -------------------- UI: mode selector --------------------
mode = st.radio("Pilih sumber video:", ("Live RTSP Stream", "Upload Video", "Live Webcam"))

# -------------------- Multi-RTSP UI --------------------
if mode == "Live RTSP Stream":
    st.subheader("🎥 Live RTSP Stream (multi 2 cameras)")

    # two-column inputs (left: cam1, right: cam2)
    col1, col2 = st.columns(2)
    with col1:
        rtsp_url_1 = st.text_input("RTSP URL Kamera 1", key="rtsp_input_1")
        start_1 = st.button("▶ Mulai Kamera 1")
        stop_1 = st.button("⏹ Hentikan Kamera 1")
        download_log_1 = st.button("📥 Download Log Kamera 1")
    with col2:
        rtsp_url_2 = st.text_input("RTSP URL Kamera 2", key="rtsp_input_2")
        start_2 = st.button("▶ Mulai Kamera 2")
        stop_2 = st.button("⏹ Hentikan Kamera 2")
        download_log_2 = st.button("📥 Download Log Kamera 2")

    # prepare per-cam log file names
    log_cam1 = "log_rtsp_cam1.csv"
    log_cam2 = "log_rtsp_cam2.csv"
    init_log_file(log_cam1)
    init_log_file(log_cam2)

    # start cam1
    if start_1 and rtsp_url_1:
        if not st.session_state.rtsp_running_1:
            st.session_state.rtsp_stop_event_1.clear()
            st.session_state.rtsp_running_1 = True
            th = threading.Thread(
                target=rtsp_worker,
                args=(
                    rtsp_url_1, 1, log_cam1,
                    RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT, RTSP_DETECT_EVERY,
                    st.session_state.rtsp_stop_event_1, st.session_state.frame_queue_1
                ),
                daemon=True,
            )
            st.session_state.rtsp_thread_1 = th
            th.start()
            write_log_csv(log_cam1, "INFO", "Start requested from UI")

    # stop cam1
    if stop_1:
        if st.session_state.rtsp_running_1:
            st.session_state.rtsp_stop_event_1.set()
            st.session_state.rtsp_running_1 = False
            write_log_csv(log_cam1, "INFO", "Stop requested from UI")

    # start cam2
    if start_2 and rtsp_url_2:
        if not st.session_state.rtsp_running_2:
            st.session_state.rtsp_stop_event_2.clear()
            st.session_state.rtsp_running_2 = True
            th = threading.Thread(
                target=rtsp_worker,
                args=(
                    rtsp_url_2, 2, log_cam2,
                    RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT, RTSP_DETECT_EVERY,
                    st.session_state.rtsp_stop_event_2, st.session_state.frame_queue_2
                ),
                daemon=True,
            )
            st.session_state.rtsp_thread_2 = th
            th.start()
            write_log_csv(log_cam2, "INFO", "Start requested from UI")

    # stop cam2
    if stop_2:
        if st.session_state.rtsp_running_2:
            st.session_state.rtsp_stop_event_2.set()
            st.session_state.rtsp_running_2 = False
            write_log_csv(log_cam2, "INFO", "Stop requested from UI")

    # download helpers
    def download_file_button(path, label):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                st.download_button(label=label, data=f, file_name=os.path.basename(path), mime="text/csv")
        else:
            st.info("Log file belum tersedia.")

    if download_log_1:
        download_file_button(log_cam1, f"Download log cam1 ({log_cam1})")
    if download_log_2:
        download_file_button(log_cam2, f"Download log cam2 ({log_cam2})")

    # display area (two columns)
    disp1, disp2 = st.columns(2)
    img1_placeholder = disp1.empty()
    fps1_placeholder = disp1.empty()
    img2_placeholder = disp2.empty()
    fps2_placeholder = disp2.empty()

    # main loop: poll frame queues and update UI
    try:
        while st.session_state.rtsp_running_1 or st.session_state.rtsp_running_2:
            # cam1
            if st.session_state.rtsp_running_1:
                try:
                    frame = st.session_state.frame_queue_1.get_nowait()
                    # frame is BGR
                    img1_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    fps1_placeholder.markdown("**Cam1 running**")
                except queue.Empty:
                    fps1_placeholder.markdown("**Cam1 running**")
            else:
                black = np.zeros((RTSP_FRAME_HEIGHT, RTSP_FRAME_WIDTH, 3), dtype=np.uint8)
                img1_placeholder.image(black, use_container_width=True)
                fps1_placeholder.markdown("**Cam1 idle**")

            # cam2
            if st.session_state.rtsp_running_2:
                try:
                    frame = st.session_state.frame_queue_2.get_nowait()
                    img2_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    fps2_placeholder.markdown("**Cam2 running**")
                except queue.Empty:
                    fps2_placeholder.markdown("**Cam2 running**")
            else:
                black = np.zeros((RTSP_FRAME_HEIGHT, RTSP_FRAME_WIDTH, 3), dtype=np.uint8)
                img2_placeholder.image(black, use_container_width=True)
                fps2_placeholder.markdown("**Cam2 idle**")

            # small sleep to reduce CPU
            time.sleep(0.05)
    except Exception as e:
        st.error(f"Display loop stopped: {e}")

# -------------------- Upload Video Mode --------------------
elif mode == "Upload Video":
    st.subheader("📁 Upload Video")
    init_log_file(LOG_UPLOAD)

    uploaded_file = st.file_uploader("Upload video (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    start_upload = st.button("▶ Mulai Pemutaran")
    stop_upload = st.button("⏹ Hentikan Pemutaran")
    download_upload_log = st.button("📥 Download Log Upload")

    # download helper (reused)
    def download_file(path, label):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                st.download_button(label=label, data=f, file_name=os.path.basename(path), mime="text/csv")
        else:
            st.info("Log file belum tersedia.")

    if download_upload_log:
        download_file(LOG_UPLOAD, "Download Log Upload")

    if uploaded_file and start_upload:
        # create local detector/tracker instances (per-run)
        detector_u = Detector()
        tracker_u = Tracker()

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.subheader("🧠 Video dengan Deteksi & Tracking")
        st.video(video_path)
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        fps_display = st.empty()

        frame_count = 0
        last_detections = np.zeros((0,5), dtype=float)

        fps_val = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps_val if fps_val > 0 else 0.03

        write_log_csv(LOG_UPLOAD, "INFO", f"Upload video dimulai: {os.path.basename(video_path)}")
        running = True

        while running and cap.isOpened():
            if stop_upload:
                running = False
                break

            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            if frame_count % UPLOAD_DETECT_EVERY == 0:
                try:
                    raw = detector_u.detect(frame_proc)
                except Exception as e:
                    raw = None
                    write_log_csv(LOG_UPLOAD, "ERROR", f"Detect error: {e}")
                detections = normalize_detections(raw)
                last_detections = detections
                write_log_csv(LOG_UPLOAD, "DETECTION", f"Frame {frame_count} - {len(detections)} objek")
            else:
                detections = last_detections

            try:
                tracks = tracker_u.update(detections, {"img_shape": frame_proc.shape, "img_size": frame_proc.shape[:2]})
            except Exception as e:
                tracks = []
                write_log_csv(LOG_UPLOAD, "ERROR", f"Tracker error: {e}")

            annotated = draw_boxes(frame_proc, tracks)

            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            fps_display.markdown(f"**FPS: {1.0 / (time.time() - start_time + 1e-5):.2f}**")

            if delay - (time.time() - start_time) > 0:
                time.sleep(delay - (time.time() - start_time))

        cap.release()
        try:
            os.remove(video_path)
        except Exception:
            pass
        write_log_csv(LOG_UPLOAD, "INFO", "Upload video dihentikan")

# -------------------- Live Webcam Mode (WebRTC) --------------------
elif mode == "Live Webcam":
    st.subheader("📷 Live Webcam (WebRTC)")
    init_log_file(LOG_WEBRTC)

    download_webcam_log = st.button("📥 Download Log Webcam")
    if download_webcam_log:
        if os.path.exists(LOG_WEBRTC) and os.path.getsize(LOG_WEBRTC) > 0:
            with open(LOG_WEBRTC, "rb") as f:
                st.download_button(label="Download Log Webcam", data=f, file_name=os.path.basename(LOG_WEBRTC), mime="text/csv")
        else:
            st.info("Log webcam belum tersedia.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            # create local detector/tracker for this processor
            self.detector = Detector()
            self.tracker = Tracker()
            self.frame_count = 0
            self.last_detections = np.zeros((0,5), dtype=float)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % 5 == 0:
                try:
                    raw = self.detector.detect(img)
                except Exception as e:
                    raw = None
                    write_log_csv(LOG_WEBRTC, "ERROR", f"Detect error: {e}")
                detections = normalize_detections(raw)
                self.last_detections = detections
                write_log_csv(LOG_WEBRTC, "DETECTION", f"Frame {self.frame_count} - {len(detections)} objek")
            else:
                detections = self.last_detections

            try:
                tracks = self.tracker.update(detections, {"img_shape": img.shape, "img_size": img.shape[:2]})
            except Exception as e:
                tracks = []
                write_log_csv(LOG_WEBRTC, "ERROR", f"Tracker error: {e}")

            annotated = draw_boxes(img, tracks)
            return annotated

    webrtc_streamer(
        key="webcam_mode",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )
