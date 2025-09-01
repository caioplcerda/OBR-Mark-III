# web_stream.py
# Flask + Socket.IO + MJPEG robusto.
# Inclui "FORCE_RGB_INPUT" para evitar cores trocadas e stream preto.

import cv2
from flask import Flask, Response, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Se a imagem sair com cores invertidas, deixe True
FORCE_RGB_INPUT = False

# Estado compartilhado; o main.py pode sobrescrever esse dict
SHARED_STATE = {
    "config": {},
    "last_frame": None,          # numpy array (BGR ou RGB com flag)
    "speeds": {"left": 0, "right": 0},
    "status": "idle",
    "view_mode": "preview",
    "fps": 0.0,
    "log": [],
}

_robot = None
_state_lock = threading.Lock()

def register_robot(robot):
    global _robot
    _robot = robot

def _get_frame():
    with _state_lock:
        return SHARED_STATE.get("last_frame", None)

# MJPEG
def mjpeg_generator():
    while True:
        frame = _get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            img = frame
            if FORCE_RGB_INPUT:
                # se pipeline estiver em RGB, converte para BGR antes de encodar
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, jpg = cv2.imencode(".jpg", img)
            if not ok:
                time.sleep(0.01)
                continue
            data = jpg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        except Exception:
            time.sleep(0.01)

# Rotas
@app.route("/")
def root():
    return send_from_directory(".", "index.html")

@app.route("/index.html")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def state():
    with _state_lock:
        return jsonify({
            "status": SHARED_STATE.get("status", ""),
            "speeds": SHARED_STATE.get("speeds", {}),
            "fps": SHARED_STATE.get("fps", 0),
            "view_mode": SHARED_STATE.get("view_mode", "preview"),
        })

# Socket.IO
@socketio.on("connect")
def on_connect():
    emit("log_message", {"text": "Socket conectado."})
    with _state_lock:
        emit("status", {
            "status": SHARED_STATE.get("status", "idle"),
            "speeds": SHARED_STATE.get("speeds", {"left": 0, "right": 0}),
            "fps": SHARED_STATE.get("fps", 0)
        })

@socketio.on("command")
def on_command(cmd):
    global _robot
    name = (cmd or {}).get("name", "")
    data = (cmd or {}).get("data", {}) or {}

    if _robot is None:
        emit("log_message", {"text": "Nenhum robô registrado ainda."})
        return

    try:
        if name == "start_robot":
            _robot.start()
            emit("log_message", {"text": "Robô iniciado."})
        elif name == "stop_robot":
            _robot.stop()
            emit("log_message", {"text": "Robô parado."})
        elif name == "set_view_mode":
            _robot.set_view_mode(data.get("mode", "preview"))
        elif name == "calibrate_pixel":
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            color = data.get("color", "black")
            ok = _robot.calibrate_pixel(x, y, color)
            emit("log_message", {"text": f"Calibração ({color}) {x},{y} -> {ok}"})
        elif name == "save_config":
            ok = _robot.save_config(data)
            emit("log_message", {"text": f"Config salva: {ok}"})
        else:
            emit("log_message", {"text": f"Comando desconhecido: {name}"})
    except Exception as e:
        emit("log_message", {"text": f"Erro no comando {name}: {e}"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
