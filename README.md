# ✨ DataSetKurator ✨

DataSetKurator turns an anime film into a dataset ready for LoRA training. Every step is logged to `logs/process.log`.

## Pipeline Overview

1. **Frame Extraction** – `ffmpeg` pulls images from the video
2. **Deduplication** – perceptual hashing drops duplicates
3. **Filtering** – remove unwanted shots
4. **Upscaling & Quality Check** – RealESRGAN or PIL resize with blur/dark checks
5. **Cropping** – faces cut out using `animeface` or a YOLOv8 model automatically detected in `models/`
6. **Annotation** – WD14 tagger generates captions
7. **Character Classification** – images grouped by detected hair and eye color
   (`<hair>_<eyes>`). Unrecognized images are placed in `unclassified`.
8. **Packaging** – images and captions are zipped for download

## Install via setup script

Run as root to install the app as a service:

```bash
sudo ./setup.sh --install
```

This copies the project to `/opt/DataSetKurator`, creates a virtual environment, installs all Python dependencies and enables a `systemd` service.

To remove the installation run:

```bash
sudo ./setup.sh --Deinstall
```

Update an existing installation with:

```bash
sudo ./setup.sh --Update
```

## Manual Run

For a local test without the service:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8000](http://localhost:8000) to upload a video and start the pipeline.
You can optionally set a *trigger word* before starting. This word will be
prepended as the first tag in every generated caption.
Place any YOLOv8 ``.pt`` file inside the new ``models/`` folder. The pipeline
indexes this directory on startup and automatically uses the first matching
weight file. If no model is found, cropping falls back to ``animeface``.
Pretrained weights such as ``AniRef40000-m-epoch75.pt`` can be obtained from the
[AniRef-yolov8 releases](https://github.com/SoulflareRC/AniRef-yolov8/releases).
