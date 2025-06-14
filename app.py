from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    jsonify,
)
import subprocess
from pathlib import Path
import shutil
from threading import Thread, Timer

from pipeline.pipeline_runner import Pipeline
from pipeline.logging_utils import LOG_FILE

INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
WORK_DIR = Path('work')

app = Flask(__name__)

status = "Idle"
progress = {"step": 0, "total": 8, "name": "Idle"}
zip_result: Path | None = None


def get_commit_id() -> str:
    """Return the short git commit hash for the current repo."""
    try:
        return (
            subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


COMMIT_ID = get_commit_id()


def schedule_zip_cleanup(path: Path, delay: int = 300) -> None:
    """Remove ``path`` after ``delay`` seconds."""

    def _delete() -> None:
        global zip_result
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if zip_result == path:
            zip_result = None

    Timer(delay, _delete).start()


def update_progress(step: int, name: str) -> None:
    """Update global progress indicator."""
    global progress
    progress["step"] = step
    progress["name"] = name

template = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>DataSetKurator</title>
  <style>
    body {background:#121212;color:#eee;font-family:Arial,sans-serif;padding:20px;}
    input,button{background:#333;color:#eee;border:1px solid #555;padding:8px;}
    button{cursor:pointer;}
    #status{margin-top:20px;font-weight:bold;}
    a{color:#4ea3ff;}
  </style>
</head>
<body>
  <h1>DataSetKurator</h1>
  <div id=\"upload-section\">
    <input type=file id=\"video-file\">
    <button id=\"upload-btn\">Upload</button>
  </div>
  <div id=\"start-section\" style=\"display:none;\">
    <input type=text id=\"trigger-word\" placeholder=\"Trigger word\" value=\"name\">
    <button id=\"start-btn\">Start Pipeline</button>
  </div>
  <div id=\"status\">Status: Idle</div>
  <div id=\"progress\" style=\"margin-top:10px;\"></div>
  <div id=\"download\" style=\"display:none;\">
    <a id=\"download-link\" href=#>Download Result</a>
  </div>
  <script>
  async function checkStatus(){
    const r = await fetch('/status');
    if(!r.ok)return;
    const d = await r.json();
    document.getElementById('status').textContent = 'Status: '+d.status;
    const prog = document.getElementById('progress');
    if(d.progress && d.progress.step){
      prog.textContent = 'Step '+d.progress.step+' / '+d.progress.total+': '+d.progress.name;
    }else{
      prog.textContent = '';
    }
    const dl = document.getElementById('download');
    const link = document.getElementById('download-link');
    if(d.status==='Completed'){
      dl.style.display='block';
      link.textContent='Download Result';
      link.href='/download';
    }else if(d.status==='Failed'){
      dl.style.display='block';
      link.textContent='Download Log';
      link.href='/log';
    }else{
      dl.style.display='none';
    }
  }
  setInterval(checkStatus,2000);
  checkStatus();

  document.getElementById('upload-btn').onclick = async () => {
    const inp = document.getElementById('video-file');
    if(!inp.files.length) return alert('Select a file');
    const fd = new FormData();
    fd.append('video', inp.files[0]);
    const r = await fetch('/upload', {method:'POST', body:fd});
    const d = await r.json();
    alert(d.message);
    document.getElementById('start-section').style.display='block';
  };

  document.getElementById('start-btn').onclick = async () => {
    const tw = document.getElementById('trigger-word').value || 'name';
    await fetch('/start', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({trigger_word: tw})
    });
  };
  </script>
  <footer style="margin-top:40px;font-size:0.8em;">Version: {{ commit_id }}</footer>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(template, commit_id=COMMIT_ID)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('video')
    if not file:
        return jsonify({'message': 'No file uploaded'}), 400
    INPUT_DIR.mkdir(exist_ok=True)
    for f in INPUT_DIR.glob('*'):
        f.unlink()
    video_path = INPUT_DIR / file.filename
    file.save(video_path)
    return jsonify({'message': f'Upload successful: {file.filename}'})

@app.route('/start', methods=['POST'])
def start():
    global status, zip_result, progress
    if status == 'Processing':
        return jsonify({'message': 'Pipeline already running'}), 400
    videos = list(INPUT_DIR.glob('*'))
    if not videos:
        return jsonify({'message': 'No videos found in input'}), 400
    video = videos[0]

    data = request.get_json(silent=True) or {}
    trigger_word = data.get('trigger_word', 'name')

    status = 'Processing'
    progress = {"step": 0, "total": 8, "name": "Queued"}
    update_progress(0, 'Queued')
    zip_result = None

    def run():
        global status, zip_result, progress
        out_dir = OUTPUT_DIR / trigger_word
        work_dir = WORK_DIR / trigger_word
        pipeline = Pipeline(INPUT_DIR, out_dir, work_dir)
        try:
            zip_result = pipeline.run(video, trigger_word=trigger_word, progress_cb=update_progress)
            schedule_zip_cleanup(Path(zip_result))
            update_progress(progress['total'], 'Done')
            status = 'Completed'
        except Exception:
            update_progress(progress['total'], 'Failed')
            status = 'Failed'
        finally:
            for f in INPUT_DIR.glob('*'):
                f.unlink()

    Thread(target=run, daemon=True).start()
    return jsonify({'message': 'Pipeline started'})

@app.route('/status')
def get_status():
    return jsonify({'status': status, 'progress': progress})

@app.route('/download')
def download():
    if zip_result and Path(zip_result).exists():
        return send_file(zip_result, as_attachment=True)
    return 'No result available', 404

@app.route('/log')
def log_file():
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
