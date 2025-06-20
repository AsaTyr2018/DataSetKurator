# 🌟 DataSetKurator  — Public Beta

**Transform any anime movie into a structured LoRA dataset with just one click.**

DataSetKurator ships a sleek web interface and an automated pipeline that turns
raw footage into neatly cropped images with captions. Every step is logged to
`logs/process.log` for full transparency.

---

## ✨ Highlights

- **Web Dashboard** with drag‑and‑drop uploads and live log view
- **Custom Settings Panel** to tune FPS, deduplication, upscaling and more
- **Model Preloading** for faster inference (YOLOv8, WD14 and RealESRGAN)
- **Automatic Version Display** derived from `changelog.md`
- **Improved Character Classification** with CLIP clustering
- **Flexible Tagging** thanks to adjustable thresholds

## 🛠 Pipeline

1. **Frame Extraction** – `ffmpeg` grabs frames from the video
2. **Deduplication** – perceptual hashing removes near duplicates
3. **Filtering** – flatten folders and drop unwanted shots
4. **Upscaling & QC** – RealESRGAN or PIL resize with blur/dark checks
5. **Cropping** – detects faces via YOLOv8, `mediapipe` or `animeface`
6. **Annotation** – WD14 tagger creates captions
7. **Character Classification** – groups images by hair/eye color, length and glasses
8. **Packaging** – outputs images and captions zipped for download

## 🚀 Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8000](http://localhost:8000) and drop your video.
Configure the settings as needed, hit **Start Batch** and watch the progress.
Each pipeline step now has its own box with a "Skip" checkbox so you can
bypass individual stages.

## ⚙️ Service Installation

Running on a server? Use the setup script:

```bash
sudo ./setup.sh --install
```

This copies everything to `/opt/DataSetKurator`, installs requirements and
creates a `systemd` service. Update or remove it via `--Update` or `--Deinstall`.

## 🙏 Credits

- [SoulflareRC/AniRef-yolov8](https://github.com/SoulflareRC/AniRef-yolov8)
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

DataSetKurator v0.25.01 Public Beta – one tool to rule them all.
