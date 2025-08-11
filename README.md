# computer-vision-test-1-AI-Engineer---Anugrah-Aidin-Yotolembah

CCTV AI Pipeline : RTSP Real-time Detection & Tracking.

Proyek ini adalah sistem deteksi objek berbasis YOLO yang berjalan melalui dashboard interaktif menggunakan Streamlit.
Mendukung empat skenario input:

1. RTSP Lokal → Menggunakan MediaMTX + OBS untuk testing kamera lokal.

2. RTSP Publik → Menggunakan URL RTSP langsung dari internet.

3. Video Upload → Mengunggah file video lokal.

4. Live Webcam → Menggunakan kamera laptop/PC secara langsung.

Semua mode dijalankan dari satu program utama: dashboard.py.

---

1. Fitur Utama
🎯 Deteksi objek real-time menggunakan YOLO.

📡 Dukungan 4 sumber input:

RTSP Lokal (MediaMTX + OBS)

RTSP Publik (URL langsung dari internet)

File video upload

Live Webcam

🔄 Auto-reconnect RTSP jika koneksi terputus.

⚡ Frame skipping untuk menghemat performa.

📝 Logging otomatis ke CSV (log_rtsp.csv, log_upload.csv, log_webcam.csv).

📊 Dashboard interaktif dengan Streamlit.

--- 

2. Persyaratan
Python 3.8 atau lebih baru

Pip sudah terpasang

MediaMTX terpasang (untuk mode RTSP Lokal)

OBS Studio terpasang (untuk menyiarkan video ke MediaMTX)

Koneksi internet (untuk mode RTSP Publik atau instalasi dependensi)

File model YOLO (.pt) sudah tersedia

---

3. instalasi 

- Clone repository
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
