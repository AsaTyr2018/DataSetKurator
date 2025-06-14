from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    jsonify,
)
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
zip_result: Path | None = None

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
    <button id=\"start-btn\">Start Pipeline</button>
  </div>
  <div id=\"status\">Status: Idle</div>
  <div id=\"download\" style=\"display:none;\">
    <a id=\"download-link\" href=#>Download Result</a>
  </div>
  <script>
  async function checkStatus(){
    const r = await fetch('/status');
    if(!r.ok)return;
    const d = await r.json();
    document.getElementById('status').textContent = 'Status: '+d.status;
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
    await fetch('/start', {method:'POST'});
  };
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(template)

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
    global status, zip_result
    if status == 'Processing':
        return jsonify({'message': 'Pipeline already running'}), 400
    videos = list(INPUT_DIR.glob('*'))
    if not videos:
        return jsonify({'message': 'No videos found in input'}), 400
    video = videos[0]

    status = 'Processing'
    zip_result = None

    def run():
        global status, zip_result
        pipeline = Pipeline(INPUT_DIR, OUTPUT_DIR, WORK_DIR)
        try:
            zip_result = pipeline.run(video)
            status = 'Completed'
        except Exception:
            status = 'Failed'
        finally:
            for f in INPUT_DIR.glob('*'):
                f.unlink()

    Thread(target=run, daemon=True).start()
    return jsonify({'message': 'Pipeline started'})

@app.route('/status')
def get_status():
    return jsonify({'status': status})

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
