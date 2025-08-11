# import cv2

# # Ganti dengan URL RTSP publik atau lokal kamu
# rtsp_url = "rtsp://rtspstream:Y9RQEg9q8hn_esZjbT_Xl@zephyr.rtsp.stream/people"

# # Buat objek VideoCapture
# cap = cv2.VideoCapture(rtsp_url)

# # Cek apakah berhasil dibuka
# if not cap.isOpened():
#     print("❌ Tidak dapat membuka RTSP stream")
#     exit()

# # Ambil ukuran frame
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# if fps == 0:
#     fps = 25  # fallback jika RTSP tidak kirim info FPS

# print(f"✅ Stream dibuka. Resolusi: {frame_width}x{frame_height}, FPS: {fps}")

# # Simpan ke file (opsional)
# save_output = True
# if save_output:
#     out = cv2.VideoWriter(
#         "output_rtsp.avi",
#         cv2.VideoWriter_fourcc(*'XVID'),
#         fps,
#         (frame_width, frame_height)
#     )

# # Loop untuk menampilkan frame
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Gagal membaca frame (stream mungkin terputus)")
#         break

#     # Tampilkan video
#     cv2.imshow("RTSP Video", frame)

#     # Simpan jika diaktifkan
#     if save_output:
#         out.write(frame)

#     # Tekan 'q' untuk keluar
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("⏹️ Dihentikan oleh user")
#         break

# # Bersihkan semuanya
# cap.release()
# if save_output:
#     out.release()
# cv2.destroyAllWindows()


import cv2
import time
import threading

# Dummy detector and tracker classes for illustration
class Detector:
    def detect(self, frame):
        # Dummy detection: return list of bounding boxes
        return [(50, 50, 100, 100)]  # x, y, w, h

class Tracker:
    def __init__(self):
        self.objects = {}

    def update(self, detections):
        # Dummy tracking logic: assign IDs to detections
        self.objects = {i: det for i, det in enumerate(detections)}
        return self.objects

class RTSPPipeline:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.detector = Detector()
        self.tracker = Tracker()
        self.cap = None
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run).start()

    def _run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print("Connecting to RTSP stream...")
                self.cap = cv2.VideoCapture(self.rtsp_url)
                time.sleep(2)  # Wait for connection

            ret, frame = self.cap.read()
            if not ret:
                print("Lost connection. Trying to reconnect...")
                self.cap.release()
                self.cap = None
                time.sleep(3)
                continue

            detections = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detections)

            self._output(frame, tracked_objects)

            # Simulate processing time for real-time optimization
            time.sleep(0.03)  # ~30 FPS

    def _output(self, frame, tracked_objects):
        # For simplicity, just print tracking info
        print(f"Tracked objects: {tracked_objects}")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    pipeline = RTSPPipeline("rtsp://localhost:8554/live/mystream")
    pipeline.start()

    # Run for 10 seconds then stop
    time.sleep(10)
    pipeline.stop()
