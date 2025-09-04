# web_stream.py
# Servidor Web + Socket.IO + MJPEG otimizado
# - /           -> index (HTML simples opcional)
# - /state      -> JSON com estado (UI lê em polling)
# - /stream     -> MJPEG com resize e JPEG quality (rápido)
# - /frame.jpg  -> snapshot único (útil para testes)
#
# Exporta:
#   - app
#   - socketio
#   - SHARED_STATE (dict mutável compartilhado com main.py)
#   - register_robot(robot)  -> opcional; main pode usá-la
#
# Observações de performance:
#  - Redimensiona para 320x240 só no stream (não mexe no frame "cheio" do vision)
#  - Qualidade JPEG 65 (ajuste p/ 50 se precisar mais leve)
#  - TurboJPEG se disponível (muito mais rápido)
#  - Sempre pega o frame MAIS RECENTE do SHARED_STATE (não cria fila)

import os
import time
import json
import threading
from datetime import datetime

from flask import Flask, Response, render_template_string, jsonify, request, make_response
from flask_cors import CORS

# Socket.IO
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except Exception:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None

# OpenCV / Numpy
import cv2
import numpy as np

# ====== Estado compartilhado com o main.py ======
SHARED_STATE = {
    "config": {},
    "last_frame": None,            # np.ndarray (BGR)
    "speeds": {"left": 0, "right": 0},
    "view_mode": "preview",        # "preview" desenha overlays
    "status": "idle",
    "fps": 0.0,
    "log": [],
}

# ====== TurboJPEG opcional ======
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
except Exception:
    _jpeg = None

# ====== App / Socket ======
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

if SOCKETIO_AVAILABLE:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",          # mantém compatível sem eventlet/gevent
        logger=False,
        engineio_logger=False,
        ping_timeout=10,
        ping_interval=10,
        max_http_buffer_size=8_000_000,
    )
else:
    socketio = None

# ====== MJPEG (rápido) ======
STREAM_W, STREAM_H = 320, 240
JPEG_QUALITY = 65  # baixar para 50 se quiser ainda mais leve

def _encode_jpeg_bgr(img_bgr: np.ndarray) -> bytes | None:
    """Codifica BGR -> JPEG. Usa TurboJPEG se disponível."""
    if img_bgr is None:
        return None
    try:
        if _jpeg is not None:
            # TurboJPEG espera RGB por padrão
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return _jpeg.encode(img_rgb, quality=JPEG_QUALITY)
        # OpenCV
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", img_bgr, enc_param)
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None

def mjpeg_generator():
    """Gera multipart/x-mixed-replace com JPEGs recentes (sem formar fila)."""
    last_sent = 0.0
    boundary = b"--frame"
    while True:
        try:
            frame = SHARED_STATE.get("last_frame", None)
            if frame is None or not isinstance(frame, np.ndarray):
                time.sleep(0.01)
                continue

            # Redimensiona só para o stream (barato e suficiente p/ UI)
            small = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)

            jpg = _encode_jpeg_bgr(small)
            if jpg is None:
                time.sleep(0.005)
                continue

            # monta bloco
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")

            # limita stream para ~25 fps (não canibalizar CPU da visão)
            now = time.time()
            dt = now - last_sent
            target = 1.0 / 25.0
            if dt < target:
                time.sleep(target - dt)
            last_sent = now

        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.02)

# ====== Rotas ======

_INDEX_HTML = """<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Robot UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; background:#0b0e11; color:#eee;}
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
    .card { background:#11151a; border:1px solid #1a2027; border-radius:12px; padding:12px; }
    .btn { background:#2563eb; color:white; border:none; padding:10px 14px; border-radius:10px; cursor:pointer; }
    .btn:disabled { background:#374151; cursor:not-allowed; }
    .stat { font-size:14px; color:#aab; }
    img { border-radius:8px; border:1px solid #1f2937; }
    pre { white-space:pre-wrap; max-height:300px; overflow:auto; background:#0f1317; border:1px solid #1a2027; padding:8px; border-radius:8px;}
    label { font-size:14px; color:#aab; margin-right:8px;}
    select { background:#0f1317; color:#eee; border:1px solid #1f2937; border-radius:8px; padding:6px 8px; }
  </style>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js" crossorigin="anonymous"></script>
</head>
<body>
  <div class="row">
    <div class="card">
      <img id="stream" src="/stream" width="640" height="480" />
      <div style="margin-top:8px;">
        <label>View:</label>
        <select id="view">
          <option value="preview">preview</option>
          <option value="raw">raw</option>
        </select>
        <button class="btn" id="start">Start</button>
        <button class="btn" id="stop" style="background:#ef4444;">Stop</button>
      </div>
      <div class="stat" id="stats"></div>
    </div>
    <div class="card" style="flex:1; min-width:320px;">
      <h3>Log</h3>
      <pre id="log"></pre>
    </div>
  </div>

  <script>
    const socket = io(); // deixa negociar websocket quando possível

    document.getElementById('start').onclick = () => socket.emit('command', { name:'start_robot' });
    document.getElementById('stop').onclick  = () => socket.emit('command', { name:'stop_robot' });

    document.getElementById('view').onchange = (e) => {
      socket.emit('command', { name:'set_view_mode', data:{ mode: e.target.value } });
    };

    async function loopState(){
      try {
        const r = await fetch('/state', { cache:'no-store' });
        const s = await r.json();
        document.getElementById('stats').textContent =
          `status: ${s.status} | fps: ${s.fps} | L:${s.speeds.left} R:${s.speeds.right} | ${s.time}`;
        const log = (s.log||[]).slice(-200).join('\\n');
        document.getElementById('log').textContent = log;
      } catch(e) {}
      setTimeout(loopState, 250); // 4 Hz de polling é suficiente
    }
    loopState();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    # HTML embutido para ser auto-suficiente
    resp = make_response(render_template_string(_INDEX_HTML))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.route("/state")
def state():
    # devolve snapshot leve (sem imagem)
    out = {
        "status": SHARED_STATE.get("status", "idle"),
        "speeds": SHARED_STATE.get("speeds", {"left":0,"right":0}),
        "fps": SHARED_STATE.get("fps", 0.0),
        "time": datetime.now().strftime("%H:%M:%S"),
        "view_mode": SHARED_STATE.get("view_mode", "preview"),
        "log": SHARED_STATE.get("log", []),
    }
    resp = jsonify(out)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.route("/frame.jpg")
def frame_jpg():
    frame = SHARED_STATE.get("last_frame", None)
    if frame is None or not isinstance(frame, np.ndarray):
        return ("", 204)
    # snapshot em 640x480 (ou no tamanho atual do frame)
    jpg = _encode_jpeg_bgr(frame)
    if jpg is None:
        return ("", 500)
    return Response(jpg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

@app.route("/stream")
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )

# ====== Socket.IO ======
_robot_ref = None
_robot_lock = threading.Lock()

def register_robot(robot_obj):
    """Permite ao main.py registrar a instância do robô para comandos via socket."""
    global _robot_ref
    with _robot_lock:
        _robot_ref = robot_obj

if SOCKETIO_AVAILABLE:
    @socketio.on("connect")
    def on_connect():
        emit("connected", {"ok": True, "ts": time.time()})

    @socketio.on("command")
    def on_command(cmd):
        # Formato esperado:
        # { name: "start_robot" | "stop_robot" | "set_view_mode" | "calibrate_pixel" | "save_config", data: {...} }
        name = None
        data = None
        try:
            name = cmd.get("name")
            data = cmd.get("data", {})
        except Exception:
            pass

        try:
            # Se o main registrou um robô, use a API dele (mais rápido/limpo)
            global _robot_ref
            r = _robot_ref
            if r is not None:
                if name == "start_robot":
                    r.start()
                    emit("ack", {"ok": True, "action": "start"})
                    return
                elif name == "stop_robot":
                    r.stop()
                    emit("ack", {"ok": True, "action": "stop"})
                    return
                elif name == "set_view_mode":
                    r.set_view_mode(str(data.get("mode", "preview")))
                    emit("ack", {"ok": True, "action": "set_view_mode"})
                    return
                elif name == "calibrate_pixel":
                    x = int(data.get("x", 0))
                    y = int(data.get("y", 0))
                    color = str(data.get("color", "green"))
                    ok = r.calibrate_pixel(x, y, color)
                    emit("ack", {"ok": bool(ok), "action": "calibrate_pixel"})
                    return
                elif name == "save_config":
                    ok = r.save_config(data or {})
                    emit("ack", {"ok": bool(ok), "action": "save_config"})
                    return

            # Fallback sem robô: altera só estado
            if name == "set_view_mode":
                SHARED_STATE["view_mode"] = str(data.get("mode", "preview"))
                emit("ack", {"ok": True, "action": "set_view_mode"})
                return

            emit("ack", {"ok": False, "error": "unknown_command_or_robot_not_registered"})
        except Exception as e:
            try:
                emit("ack", {"ok": False, "error": str(e)})
            except Exception:
                pass

# ====== Exec direto (debug) ======
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print(f"[web_stream] up at http://{host}:{port}", flush=True)
    if socketio is not None:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    else:
        app.run(host=host, port=port)
