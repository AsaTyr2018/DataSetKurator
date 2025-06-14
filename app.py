from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    jsonify,
)
import subprocess
import time
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
zip_expire: float | None = None


REPO_ROOT = Path(__file__).resolve().parent


def get_commit_id() -> str:
    """Return the short git commit hash for the repository.

    Falls back to ``unknown`` if ``git`` is unavailable or the repo is missing.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


COMMIT_ID = get_commit_id()


def schedule_zip_cleanup(path: Path, delay: int = 300) -> None:
    """Remove ``path`` after ``delay`` seconds and track its expiry."""

    global zip_expire
    zip_expire = time.time() + delay

    def _delete() -> None:
        global zip_result, zip_expire
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if zip_result == path:
            zip_result = None
        zip_expire = None

    Timer(delay, _delete).start()


def update_progress(step: int, name: str) -> None:
    """Update global progress indicator."""
    global progress
    progress["step"] = step
    progress["name"] = name

template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DataSetKurator</title>
  <style>
    body{background:#121212;color:#eee;font-family:Arial,sans-serif;margin:0;padding:20px;display:flex;flex-direction:column;align-items:center;}
    #drop-zone{border:2px dashed #555;padding:40px;width:100%;max-width:400px;text-align:center;margin-bottom:20px;cursor:pointer;}
    #drop-zone.hover{border-color:#4ea3ff;}
    input,button{background:#333;color:#eee;border:1px solid #555;padding:8px;}
    button{cursor:pointer;}
    #progress-bar{width:100%;background:#333;margin-top:10px;height:20px;display:none;max-width:400px;}
    #progress-bar .bar{height:100%;width:0;background:#4ea3ff;}
    #status{margin-top:20px;font-weight:bold;}
    a{color:#4ea3ff;}
    #ttl{margin-top:10px;}
  </style>
</head>
<body>
  <h1>DataSetKurator</h1>
  <div id="drop-zone">Drop video here or click to select</div>
  <input type="file" id="video-file" style="display:none;">
  <div id="progress-bar"><div class="bar"></div></div>
  <div id="start-section" style="display:none;">
    <input type="text" id="trigger-word" placeholder="Trigger word" value="name">
    <button id="start-btn">Start Pipeline</button>
  </div>
  <div id="status">Status: Idle</div>
  <div id="progress" style="margin-top:10px;"></div>
  <div id="download" style="display:none;">
    <a id="download-link" href="#">Download Result</a>
    <div id="ttl" style="display:none;">Expires in <span id="ttl-val">0</span>s</div>
  </div>
  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-file');
    dropZone.onclick = () => fileInput.click();
    dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add('hover'); };
    dropZone.ondragleave = () => dropZone.classList.remove('hover');
    dropZone.ondrop = e => {
        e.preventDefault();
        dropZone.classList.remove('hover');
        if(e.dataTransfer.files.length){
            fileInput.files = e.dataTransfer.files;
            uploadFile(e.dataTransfer.files[0]);
        }
    };
    fileInput.onchange = () => { if(fileInput.files.length) uploadFile(fileInput.files[0]); };

    function uploadFile(file){
        const xhr = new XMLHttpRequest();
        const bar = document.querySelector('#progress-bar .bar');
        document.getElementById('progress-bar').style.display='block';
        xhr.upload.onprogress = e => {
            if(e.lengthComputable){
                bar.style.width = (e.loaded / e.total * 100) + '%';
            }
        };
        xhr.onload = () => {
            document.getElementById('progress-bar').style.display='none';
            const resp = JSON.parse(xhr.responseText);
            alert(resp.message);
            document.getElementById('start-section').style.display='block';
        };
        const fd = new FormData();
        fd.append('video', file);
        xhr.open('POST','/upload');
        xhr.send(fd);
    }

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
      const ttl = document.getElementById('ttl');
      if(d.status==='Completed'){
        dl.style.display='block';
        link.textContent='Download Result';
        link.href='/download';
        if(d.ttl!==null){
          ttl.style.display='block';
          document.getElementById('ttl-val').textContent=d.ttl;
        }else{
          ttl.style.display='none';
        }
      }else if(d.status==='Failed'){
        dl.style.display='block';
        link.textContent='Download Log';
        link.href='/log';
        ttl.style.display='none';
      }else{
        dl.style.display='none';
        ttl.style.display='none';
      }
    }
    setInterval(checkStatus,2000);
    checkStatus();

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
    ttl = None
    if zip_expire is not None:
        ttl = max(0, int(zip_expire - time.time()))
    return jsonify({'status': status, 'progress': progress, 'ttl': ttl})

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
