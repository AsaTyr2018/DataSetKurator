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
from threading import Thread

from pipeline.pipeline_runner import Pipeline
from pipeline.logging_utils import LOG_FILE

INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
WORK_DIR = Path('work')

app = Flask(__name__)

status = "Idle"
progress = {"step": 0, "total": 8, "name": "Idle"}
zip_result: Path | None = None
results: list[str] = []


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
    body{background:#121212;color:#eee;font-family:Arial,sans-serif;margin:0;display:flex;flex-direction:column;height:100vh;}
    header{background:#1e1e1e;padding:20px;text-align:center;box-shadow:0 0 10px #000;}
    header h1{margin:0;font-size:2em;color:#4ea3ff;text-shadow:0 0 5px #4ea3ff;}
    #content{flex:1;display:flex;gap:20px;padding:20px;box-sizing:border-box;}
    .box{background:#1e1e1e;padding:20px;flex:1;border:1px solid #333;display:flex;flex-direction:column;}
    #drop-zone{border:2px dashed #555;padding:40px;text-align:center;cursor:pointer;margin-top:10px;}
    #drop-zone.hover{border-color:#4ea3ff;}
    input,button{background:#333;color:#eee;border:1px solid #555;padding:8px;}
    button{cursor:pointer;margin-top:10px;}
    #progress-bar{width:100%;background:#333;margin-top:10px;height:20px;display:none;}
    #progress-bar .bar{height:100%;width:0;background:#4ea3ff;}
    #status{font-weight:bold;margin-top:10px;}
    a{color:#4ea3ff;}
    pre{background:#000;color:#0f0;padding:10px;height:200px;overflow:auto;margin-top:10px;flex:1;}
    footer{background:#1e1e1e;padding:10px;text-align:center;font-size:0.9em;}
  </style>
</head>
<body>
  <header>
    <h1>DataSetKurator</h1>
  </header>
  <div id=\"content\">
    <div class=\"box\">
      <h2>Job Queue</h2>
      <ul id=\"queue-list\"></ul>
    </div>
    <div class=\"box\">
      <h2>Current Job</h2>
      <div id=\"status\">Status: Idle</div>
      <div id=\"progress\"></div>
      <a id=\"log-download\" href=\"/log\" target=\"_blank\">Download Log</a>
      <pre id=\"log-content\"></pre>
      <div id=\"drop-zone\">Click or Drop video to upload</div>
      <input type=\"file\" id=\"video-file\" style=\"display:none;\" multiple>
      <div id=\"progress-bar\"><div class=\"bar\"></div></div>
      <button id=\"start-btn\" style=\"display:none;\">Start Batch</button>
    </div>
    <div class=\"box\">
      <h2>Finished Jobs</h2>
      <ul id=\"result-list\"></ul>
    </div>
  </div>
  <footer>
    <div>One tool to Rule them all</div>
    <div>Presented by AsaTyr</div>
    <div>Version: {{ commit_id }}</div>
  </footer>
  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-file');
    const logBox = document.getElementById('log-content');
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
            document.getElementById('start-btn').style.display='block';
        };
        const fd = new FormData();
        fd.append('video', file);
        xhr.open('POST','/upload');
        xhr.send(fd);
    }

    async function fetchLog(){
      const r = await fetch('/log');
      if(r.ok){
        const text = await r.text();
        logBox.textContent = text;
        logBox.scrollTop = logBox.scrollHeight;
      }
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
      const list = document.getElementById('queue-list');
      list.innerHTML = '';
      for(const v of d.queue){
        const li = document.createElement('li');
        li.textContent = v;
        list.appendChild(li);
      }
      const resList = document.getElementById('result-list');
      resList.innerHTML = '';
      for(const z of d.results){
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.textContent = z;
        a.href = '/download/'+encodeURIComponent(z);
        li.appendChild(a);
        resList.appendChild(li);
      }
      const startBtn = document.getElementById('start-btn');
      if(d.queue.length && d.status!=='Processing'){
        startBtn.style.display='block';
      }else{
        startBtn.style.display='none';
      }
    }

    setInterval(() => {checkStatus(); fetchLog();}, 2000);
    checkStatus();
    fetchLog();

    document.getElementById('start-btn').onclick = async () => {
      await fetch('/start', {method:'POST'});
    };
  </script>
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
    video_path = INPUT_DIR / file.filename
    file.save(video_path)
    return jsonify({'message': f'Upload successful: {file.filename}'})

@app.route('/start', methods=['POST'])
def start():
    global status, zip_result, progress, results
    if status == 'Processing':
        return jsonify({'message': 'Pipeline already running'}), 400
    videos = sorted(INPUT_DIR.glob('*'))
    if not videos:
        return jsonify({'message': 'No videos found in input'}), 400

    status = 'Processing'
    progress = {"step": 0, "total": 8, "name": "Queued"}
    update_progress(0, 'Queued')
    zip_result = None

    def run():
        global status, zip_result, progress, results
        try:
            for video in videos:
                tw = video.stem
                out_dir = OUTPUT_DIR / tw
                work_dir = WORK_DIR / tw
                pipeline = Pipeline(INPUT_DIR, out_dir, work_dir)
                zip_result = pipeline.run(video, trigger_word=tw, progress_cb=update_progress)
                results.append(Path(zip_result).name)
                video.unlink()
            update_progress(progress['total'], 'Done')
            status = 'Completed'
        except Exception:
            update_progress(progress['total'], 'Failed')
            status = 'Failed'

    Thread(target=run, daemon=True).start()
    return jsonify({'message': 'Pipeline started'})

@app.route('/status')
def get_status():
    queue = [p.name for p in sorted(INPUT_DIR.glob('*'))]
    res_names = [p for p in results]
    return jsonify({'status': status, 'progress': progress, 'queue': queue, 'results': res_names})

@app.route('/download/<name>')
def download(name: str):
    path = OUTPUT_DIR / f"{name}"
    if path.exists():
        return send_file(path, as_attachment=True)
    return 'No result available', 404

@app.route('/log')
def log_file():
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
