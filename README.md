# DataSetKurator

A minimal pipeline to convert an uploaded video into a basic dataset. Steps are logged to `logs/process.log`.

## Usage

1. Install dependencies
   ```bash
   pip install flask pillow imagehash torch open_clip_torch scikit-learn
   ```
   Ensure `ffmpeg` is installed and available in your `PATH`.
   On Debian-based systems:
   ```bash
   sudo apt-get install ffmpeg
   ```
2. Run the web app
   ```bash
   python app.py
   ```
3. Open `http://localhost:8000` in your browser

Upload a video, start the pipeline and download the resulting zip file.

The second stage of the pipeline removes near-duplicate frames using
perceptual hashing so that only unique images are kept for classification.
The third stage uses an anime-focused CLIP model and DBSCAN clustering to
automatically group frames by character.
