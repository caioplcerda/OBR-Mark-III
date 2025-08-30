# web_stream.py
# Flask + Socket.IO com estado compartilhado e streaming MJPEG do last_frame.

import cv2
from flask import Flask, Response, request, render_template_string, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Estado compartilhado (será substituído pelo main.py se ele importar e usar)
SHARED_STATE = {
    "config": {},
    "last_frame": None,          # numpy BGR
    "speeds": {"left": 0, "right": 0},
    "status": "idle",
    "view_mode": "normal",
    "derivative_scan": None,
    "path_history": [],
    "log": [],
    "fps": 0.0,
}

_robot = None
_state_lock = threading.Lock()

def register_robot(robot):
    global _robot
    _robot = robot

# ---- Helpers ----
def _get_frame_bgr():
    with _state_lock:
        return SHARED_STATE.get("last_frame", None)

def _log(msg):
    with _state_lock:
        SHARED_STATE["log"].append(msg)
        SHARED_STATE["log"] = SHARED_STATE["log"][-300:]

# ---- MJPEG stream ----
def mjpeg_generator():
    while True:
        frame = _get_frame_bgr()
        if frame is None:
            # antes do robô iniciar, aguarda um pouco
            time.sleep(0.05)
            continue
        try:
            # codifica BGR -> JPEG
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.01)
                continue
            data = jpg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        except Exception:
            time.sleep(0.01)

# ---- Rotas HTTP ----
@app.route("/")
def root():
    # se você já tem um index.html na pasta do projeto, sirva-o:
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
    # endpoint útil para debug da UI
    with _state_lock:
        return jsonify({
            "status": SHARED_STATE.get("status", ""),
            "speeds": SHARED_STATE.get("speeds", {}),
            "fps": SHARED_STATE.get("fps", 0),
            "view_mode": SHARED_STATE.get("view_mode", "normal"),
        })

# ---- Socket.IO ----
@socketio.on("connect")
def on_connect():
    emit("log_message", {"text": "Socket conectado."})
    # envia um snapshot de status
    with _state_lock:
        emit("status", {
            "status": SHARED_STATE.get("status", "idle"),
            "speeds": SHARED_STATE.get("speeds", {"left": 0, "right": 0}),
            "fps": SHARED_STATE.get("fps", 0)
        })

@socketio.on("command")
def on_command(cmd):
    """
    Espera payloads como:
      {"name":"start_robot"}
      {"name":"stop_robot"}
      {"name":"set_view_mode","data":{"mode":"mask"}}
      {"name":"calibrate_pixel","data":{"x":123,"y":321,"color":"green"}}
      {"name":"save_config","data":{"vision":{...},"pid":{...}}}
    """
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
            _robot.set_view_mode(data.get("mode", "normal"))
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
        emit("log_message", {"text": f"Erro ao executar comando {name}: {e}"})


# ---- Utilitário chamado periodicamente pelo main (opcional) ----
def update_stream_data(frame, mask, contours, speeds, status_data, derivative_data):
    """
    Se o seu main quiser empurrar dados por aqui, mantenho compatibilidade.
    A versão atual do main atualiza SHARED_STATE diretamente, então isso é opcional.
    """
    with _state_lock:
        SHARED_STATE["last_frame"] = frame
        SHARED_STATE["speeds"] = speeds or SHARED_STATE["speeds"]
        if status_data and "status" in status_data:
            SHARED_STATE["status"] = status_data["status"]
        if derivative_data is not None:
            SHARED_STATE["derivative_scan"] = derivative_data


if __name__ == "__main__":
    # útil se quiser testar este arquivo isolado (sem o main)
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
