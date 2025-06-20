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
from pipeline.logging_utils import LOG_FILE, rotate_log

INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
WORK_DIR = Path('work')

app = Flask(__name__)

status = "Idle"
progress = {"step": 0, "total": 8, "name": "Idle"}
zip_result: Path | None = None
results: list[str] = []


REPO_ROOT = Path(__file__).resolve().parent


def get_version_from_changelog(commit_id: str) -> str:
    """Return the version string for the given commit ID from changelog.md.

    If no matching entry is found, the closest ancestor commit listed in the
    changelog is used. Returns ``"Unknown"`` if the file is missing or no entry
    matches.
    """
    changelog = REPO_ROOT / "changelog.md"
    if not changelog.exists():
        return "Unknown"

    entries: list[tuple[str, str]] = []
    current_version = "Unknown"
    for line in changelog.read_text().splitlines():
        line = line.strip()
        if line.startswith("## "):
            current_version = line[3:].strip()
        elif line.startswith("-") and "(" in line and ")" in line:
            cid = line[line.rfind("(") + 1 : line.rfind(")")].strip()
            if cid:
                entries.append((cid, current_version))

    # Iterate entries in order and return the first commit that is an ancestor
    for cid, version in entries:
        try:
            subprocess.check_call(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "merge-base",
                    "--is-ancestor",
                    cid,
                    "HEAD",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return version
        except subprocess.CalledProcessError:
            continue

    return "Unknown"


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
VERSION = get_version_from_changelog(COMMIT_ID)


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
    pre{background:#000;color:#0f0;padding:10px;height:200px;max-height:800px;overflow:auto;margin-top:10px;flex:1;white-space:pre-wrap;word-break:break-word;}
    footer{background:#1e1e1e;padding:10px;text-align:center;font-size:0.9em;}
    #settings{background:#1e1e1e;color:#eee;padding:10px;border-bottom:1px solid #333;}
    #settings summary{cursor:pointer;font-weight:bold;font-size:1.2em;}
    .step-box{border:1px solid #333;padding:10px;margin-top:10px;}
    .step-box[open]{display:flex;flex-wrap:wrap;gap:10px;}
    .step-box summary{cursor:pointer;font-weight:bold;font-size:1em;margin:0;}
    .step-box[open] summary{width:100%;margin-bottom:10px;}
    .step-box label{display:flex;flex-direction:column;flex:1 1 200px;font-size:0.9em;}
    .step-box label span{margin-top:4px;}
    .step-box label small{color:#aaa;font-size:0.8em;}
  </style>
</head>
<body>
  <header>
    <h1>DataSetKurator</h1>
  </header>
  <details id=\"settings\" style=\"width:100%;\">
    <summary>Settings</summary>
    <details class=\"step-box\">
      <summary>Frame Extraction</summary>
      <label>FPS
        <input type=\"range\" id=\"fps\" min=\"1\" max=\"30\" value=\"1\" step=\"1\">
        <span id=\"fps-val\">1</span>
        <small>frames per second</small>
      </label>
      <label><input type=\"checkbox\" id=\"skip-extract\"> Skip Extraction</label>
    </details>
    <details class=\"step-box\">
      <summary>Deduplication</summary>
      <label>Dedup Threshold
        <input type=\"range\" id=\"dedup\" min=\"1\" max=\"16\" value=\"8\" step=\"1\">
        <span id=\"dedup-val\">8</span>
        <small>max hash distance</small>
      </label>
      <label><input type=\"checkbox\" id=\"skip-dedup\"> Skip Deduplication</label>
    </details>
    <details class=\"step-box\">
      <summary>Filtering</summary>
      <label><input type=\"checkbox\" id=\"skip-filter\"> Skip Filtering</label>
    </details>
    <details class=\"step-box\">
      <summary>Upscaling</summary>
      <label>Upscale
        <input type=\"range\" id=\"scale\" min=\"1\" max=\"8\" value=\"4\" step=\"1\">
        <span id=\"scale-val\">4</span>
        <small>enlarge factor</small>
      </label>
      <label>Blur Limit
        <input type=\"range\" id=\"blur\" min=\"0\" max=\"300\" value=\"100\" step=\"1\">
        <span id=\"blur-val\">100</span>
        <small>min blur variance</small>
      </label>
      <label>Dark Limit
        <input type=\"range\" id=\"dark\" min=\"0\" max=\"100\" value=\"40\" step=\"1\">
        <span id=\"dark-val\">40</span>
        <small>min brightness</small>
      </label>
      <label><input type=\"checkbox\" id=\"skip-upscale\"> Skip Upscaling</label>
    </details>
    <details class=\"step-box\">
      <summary>Cropping</summary>
      <label>Margin
        <input type=\"range\" id=\"margin\" min=\"0\" max=\"1\" value=\"0.3\" step=\"0.01\">
        <span id=\"margin-val\">0.3</span>
        <small>face crop border</small>
      </label>
      <label>YOLO Confidence
        <input type=\"range\" id=\"conf\" min=\"0\" max=\"1\" value=\"0.5\" step=\"0.01\">
        <span id=\"conf-val\">0.5</span>
        <small>min detection score</small>
      </label>
      <label>Batch Size
        <input type=\"range\" id=\"batch\" min=\"1\" max=\"16\" value=\"4\" step=\"1\">
        <span id=\"batch-val\">4</span>
        <small>images per batch</small>
      </label>
      <label><input type=\"checkbox\" id=\"skip-crop\"> Skip Cropping</label>
    </details>
    <details class=\"step-box\">
      <summary>Annotation</summary>
      <label>Trigger Word
        <input type=\"text\" id=\"trigger\" placeholder=\"video name\">
      </label>
      <label><input type=\"checkbox\" id=\"skip-annot\"> Skip Annotation</label>
    </details>
    <details class=\"step-box\">
      <summary>Classification</summary>
      <label><input type=\"checkbox\" id=\"skip-class\"> Skip Classification</label>
    </details>
  </details>
  <div id=\"content\">
    <div class=\"box\">
      <h2>Job Queue</h2>
      <div id=\"drop-zone\">Click or Drop video to upload</div>
      <input type=\"file\" id=\"video-file\" style=\"display:none;\" multiple>
      <div id=\"progress-bar\"><div class=\"bar\"></div></div>
      <button id=\"start-btn\" style=\"display:none;\">Start Batch</button>
      <ul id=\"queue-list\"></ul>
    </div>
    <div class=\"box\">
      <h2>Current Job</h2>
      <div id=\"status\">Status: Idle</div>
      <div id=\"progress\"></div>
      <a id=\"log-download\" href=\"/log\" target=\"_blank\">Download Log</a>
      <pre id=\"log-content\"></pre>
    </div>
    <div class=\"box\">
      <h2>Finished Jobs</h2>
      <ul id=\"result-list\"></ul>
    </div>
  </div>
  <footer>
    <div>One tool to Rule them all</div>
    <div>Presented by AsaTyr</div>
    <div>Version: {{ version }} ({{ commit_id }})</div>
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
        for (const file of e.dataTransfer.files){
            uploadFile(file);
        }
    };
    fileInput.onchange = () => {
        for (const file of fileInput.files){
            uploadFile(file);
        }
    };

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
        const lines = text.trim().split('\\n').reverse();
        logBox.textContent = lines.join('\\n');
        logBox.scrollTop = 0;
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

    function getSettings(){
      return {
        trigger_word: document.getElementById('trigger').value,
        fps: parseInt(document.getElementById('fps').value),
        dedup_threshold: parseInt(document.getElementById('dedup').value),
        scale: parseInt(document.getElementById('scale').value),
        blur_threshold: parseFloat(document.getElementById('blur').value),
        dark_threshold: parseFloat(document.getElementById('dark').value),
        margin: parseFloat(document.getElementById('margin').value),
        conf_threshold: parseFloat(document.getElementById('conf').value),
        batch_size: parseInt(document.getElementById('batch').value),
        skip_extraction: document.getElementById('skip-extract').checked,
        skip_deduplication: document.getElementById('skip-dedup').checked,
        skip_filtering: document.getElementById('skip-filter').checked,
        skip_upscaling: document.getElementById('skip-upscale').checked,
        skip_cropping: document.getElementById('skip-crop').checked,
        skip_annotation: document.getElementById('skip-annot').checked,
        skip_classification: document.getElementById('skip-class').checked
      };
    }

    const pairs = [
      ['fps','fps-val'],
      ['dedup','dedup-val'],
      ['scale','scale-val'],
      ['blur','blur-val'],
      ['dark','dark-val'],
      ['margin','margin-val'],
      ['conf','conf-val'],
      ['batch','batch-val']
    ];
    for(const [id,val] of pairs){
      const s = document.getElementById(id);
      const o = document.getElementById(val);
      s.oninput = () => { o.textContent = s.value; };
    }

    document.getElementById('start-btn').onclick = async () => {
      const settings = getSettings();
      await fetch('/start', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(settings)
      });
    };
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        template, commit_id=COMMIT_ID, version=VERSION
    )

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

    data = request.get_json(silent=True) or {}

    fps = int(data.get('fps', 1))
    dedup_threshold = int(data.get('dedup_threshold', 8))
    scale = int(data.get('scale', 4))
    blur_threshold = float(data.get('blur_threshold', 100.0))
    dark_threshold = float(data.get('dark_threshold', 40.0))
    margin = float(data.get('margin', 0.3))
    conf_threshold = float(data.get('conf_threshold', 0.5))
    batch_size = int(data.get('batch_size', 4))
    skip_extraction = bool(data.get('skip_extraction'))
    skip_deduplication = bool(data.get('skip_deduplication'))
    skip_filtering = bool(data.get('skip_filtering'))
    skip_upscaling = bool(data.get('skip_upscaling'))
    skip_cropping = bool(data.get('skip_cropping'))
    skip_annotation = bool(data.get('skip_annotation'))
    skip_classification = bool(data.get('skip_classification'))

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
                try:
                    zip_result = pipeline.run(
                        video,
                        trigger_word=data.get('trigger_word', tw) or tw,
                        progress_cb=update_progress,
                        fps=fps,
                        dedup_threshold=dedup_threshold,
                        scale=scale,
                        blur_threshold=blur_threshold,
                        dark_threshold=dark_threshold,
                        margin=margin,
                        conf_threshold=conf_threshold,
                        batch_size=batch_size,
                        skip_deduplication=skip_deduplication,
                        skip_filtering=skip_filtering,
                        skip_upscaling=skip_upscaling,
                        skip_cropping=skip_cropping,
                        skip_annotation=skip_annotation,
                        skip_classification=skip_classification,
                    )
                    results.append(Path(zip_result).name)
                    video.unlink()
                finally:
                    rotate_log(tw)
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
