from flask import Flask, request, render_template_string, send_file
from pathlib import Path
import shutil

from pipeline.pipeline_runner import Pipeline
from pipeline.logging_utils import LOG_FILE

INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
WORK_DIR = Path('work')

app = Flask(__name__)

template = """
<!doctype html>
<title>DataSetKurator</title>
<h1>Upload Video</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=video>
  <input type=submit value=Upload>
</form>
<form method=post action="/start">
  <button type=submit>Start Pipeline</button>
</form>
"""

@app.route('/')
def index():
    return render_template_string(template)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if not file:
        return 'No file uploaded', 400
    INPUT_DIR.mkdir(exist_ok=True)
    video_path = INPUT_DIR / file.filename
    file.save(video_path)
    return 'Upload successful: ' + file.filename

@app.route('/start', methods=['POST'])
def start():
    videos = list(INPUT_DIR.glob('*'))
    if not videos:
        return 'No videos found in input', 400
    video = videos[0]
    pipeline = Pipeline(INPUT_DIR, OUTPUT_DIR, WORK_DIR)
    try:
        zip_path = pipeline.run(video)
    except Exception:
        return send_file(LOG_FILE, as_attachment=True)
    finally:
        # clean input
        for f in INPUT_DIR.glob('*'):
            f.unlink()
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
