# web_stream.py
# Servidor Flask + Socket.IO para o segue-linha.
# - expõe: app, socketio, SHARED_STATE, register_robot(robot)
# - o main.py é quem dá socketio.run(app, ...)

import base64
import threading
from collections import deque

from flask import Flask, Response, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit

# ===== Estado compartilhado com o main =====
SHARED_STATE = {
    "config": {},
    "last_frame": None,   # numpy BGR (o main atualiza)
    "speeds": {"left": 0, "right": 0},
    "view_mode": "preview",
    "status": "idle",
    "fps": 0.0,
    "log": deque(maxlen=300),
}

# ===== Flask / SocketIO =====
app = Flask(__name__)
app.config["SECRET_KEY"] = "obr-robot"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

_robot = None
_state_lock = threading.Lock()

def register_robot(robot):
    global _robot
    _robot = robot

# ====== UI mínima (usa SHARED_STATE do main) ======
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OBR — Segue Linha</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; }
    .row { display:flex; gap:16px; align-items:center; }
    button { padding: 8px 12px; }
    img { max-width: 96vw; height:auto; background:#111; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  </style>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
</head>
<body>
  <h2>OBR — Segue Linha</h2>
  <div class="row">
    <button onclick="sendCmd('start_robot')">Start</button>
    <button onclick="sendCmd('stop_robot')">Stop</button>
    <select id="mode" onchange="setMode()">
      <option value="preview" selected>Preview</option>
      <option value="raw">Raw</option>
    </select>
    <span id="stat" class="mono"></span>
  </div>
  <p><img id="v" src="/stream" /></p>
  <pre id="log" class="mono"></pre>

<script>
const sio = io();
function sendCmd(name, data={}) { sio.emit('command', {name, data}); }
function setMode(){ const m=document.getElementById('mode').value; sendCmd('set_view_mode', {mode:m}); }

async function loopState(){
  try{
    const r = await fetch('/state');
    const s = await r.json();
    document.getElementById('stat').textContent =
      `status=${s.status}  fps=${s.fps}  L=${s.speeds.left} R=${s.speeds.right}`;
    const logEl = document.getElementById('log');
    logEl.textContent = (s.log||[]).join('\\n');
  }catch(e){}
  setTimeout(loopState, 500);
}
loopState();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/state")
def state():
    with _state_lock:
        # deques não são JSON-serializáveis diretamente
        log_list = list(SHARED_STATE.get("log", []))
        d = dict(SHARED_STATE)
        d["log"] = log_list
        # numpy arrays não podem ir aqui; a imagem vai por /stream
        d["last_frame"] = None
    return jsonify(d)

def _encode_jpeg(bgr):
    # BGR(numpy) -> JPEG bytes
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else b""

@app.route("/stream")
def stream():
    def gen():
        import time
        boundary = "--frame"
        while True:
            frame = None
            with _state_lock:
                frame = SHARED_STATE.get("last_frame", None)
            if frame is not None:
                jpg = _encode_jpeg(frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)  # ~33fps alvo (limitado pelo main)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Eventos básicos do Socket.IO (o main registra o handler de 'command')
@socketio.on("connect")
def on_connect():
    emit("hello", {"ok": True})

# Nada de socketio.run aqui — o main.py faz isso.
