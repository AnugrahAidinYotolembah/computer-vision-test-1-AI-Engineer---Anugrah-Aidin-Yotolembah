# video_processor.py
import av
import cv2
import time
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from detector import Detector
from tracker import Tracker
from rtsp_handler import RTSPStream
from utils import draw_boxes
from config import (RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT, RTSP_DETECT_EVERY,
                    FRAME_WIDTH, FRAME_HEIGHT, UPLOAD_DETECT_EVERY)

class VideoProcessor(VideoProcessorBase):
    """
    VideoProcessor that:
    - If rtsp_url provided, reads frames from RTSPStream (background reader)
    - Else uses incoming frames from browser (if any)
    - Runs detection every N frames (frame skipping), updates tracker, draws boxes
    """
    def __init__(self, rtsp_url: str = None, mode: str = "rtsp"):
        self.mode = mode  # "rtsp" or "upload"
        self.detector = Detector()
        self.tracker = Tracker()
        self.frame_count = 0
        self.last_detections = []
        self.rtsp = None
        self.running = True

        if self.mode == "rtsp":
            self.width = RTSP_FRAME_WIDTH
            self.height = RTSP_FRAME_HEIGHT
            self.detect_every = RTSP_DETECT_EVERY
            if rtsp_url:
                # instantiate background RTSP reader
                self.rtsp = RTSPStream(rtsp_url)
        else:
            self.width = FRAME_WIDTH
            self.height = FRAME_HEIGHT
            self.detect_every = UPLOAD_DETECT_EVERY

    def _get_frame_from_source(self, input_frame):
        # 1) Try RTSP frame first
        if self.rtsp:
            try:
                frm = self.rtsp.read()
                if frm is not None:
                    return frm
            except Exception:
                pass

        # 2) Then try incoming browser frame
        if input_frame is not None:
            try:
                img = input_frame.to_ndarray(format="bgr24")
                return img
            except Exception:
                return None

        return None

    def recv(self, frame):
        """
        Called by streamlit-webrtc. Must return av.VideoFrame.
        """
        if not self.running:
            # return black frame to keep connection alive
            black = np.zeros((self.height, self.width, 3), dtype="uint8")
            return av.VideoFrame.from_ndarray(black, format="bgr24")

        src = self._get_frame_from_source(frame)

        # If nothing available yet -> if input frame present return it, else black
        if src is None:
            if frame is not None:
                return frame
            black = np.zeros((self.height, self.width, 3), dtype="uint8")
            return av.VideoFrame.from_ndarray(black, format="bgr24")

        # Resize/normalize for processing
        try:
            proc = cv2.resize(src, (self.width, self.height))
        except Exception:
            proc = src

        self.frame_count += 1

        # Detection every N frames
        if self.frame_count % self.detect_every == 0:
            try:
                detections = self.detector.detect(proc)
            except Exception:
                detections = []
            self.last_detections = detections
        else:
            detections = self.last_detections

        # Update tracker
        try:
            tracks = self.tracker.update(detections, {"img_shape": proc.shape, "img_size": proc.shape[:2]})
        except Exception:
            tracks = []

        # Draw boxes
        try:
            annotated = draw_boxes(proc, tracks)
        except Exception:
            annotated = proc

        # Convert to av.VideoFrame and return
        try:
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception:
            # fallback: convert to RGB then frame
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

    def stop(self):
        self.running = False
        if self.rtsp:
            try:
                self.rtsp.stop()
            except Exception:
                pass

    def __del__(self):
        self.stop()
