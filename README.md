# DataSetKurator

A minimal pipeline to convert an uploaded video into a basic dataset. Steps are logged to `logs/process.log`.

## Usage

1. Install dependencies
   ```bash
   pip install flask pillow imagehash torch open_clip_torch scikit-learn \
       animeface numpy
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
The cropping stage relies on the ``animeface`` library to detect faces and
produce crops around them, keeping the entire image when no face is found.
The annotation stage applies the *WD14* tagger to each cropped image and
generates a comma-separated list of tags, falling back to a simple
``anime_style`` caption if the model cannot be loaded.
