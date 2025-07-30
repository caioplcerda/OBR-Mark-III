import cv2
import numpy as np
import os
import json
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# --- Dicionário de Estado Global Compartilhado ---
SHARED_STATE = {
    "config": {
        "pid": {"kp": 0.4, "ki": 0.0, "kd": 0.1},
        "hsv_black": { "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50]) }
    },
    "start_event": threading.Event(),
    "stream_lock": threading.Lock(),
    "stream_data": {
        "last_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "last_mask": np.zeros((480, 640), dtype=np.uint8),
        "path_history": [],
        "motor_speeds": {"left": 0, "right": 0},
        "status_data": {},
        "view_mode": 'normal'
    },
    "calibration_request": None
}

# --- Funções de Logging ---
def log(message):
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)  # Mantém o log no console do servidor
    socketio.emit('log_message', {'data': full_message})

# --- Rotas Flask ---
@app.route('/')
def index():
    return render_template_string(open('index.html').read(), config=SHARED_STATE['config'])

@app.route('/stream')
def stream_route():
    def generate():
        while True:
            try:
                with SHARED_STATE['stream_lock']:
                    s_data = SHARED_STATE['stream_data']
                    frame = s_data['last_frame'].copy()
                    mask = s_data['last_mask'].copy()
                    view_mode = s_data['view_mode']
                    path = list(s_data['path_history'])
                    status = dict(s_data['status_data'])
                    speeds = dict(s_data['motor_speeds'])

                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                if view_mode == 'mask':
                    output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    output_frame = frame
                    if view_mode == 'contours' and mask is not None and mask.size > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)

                    if path:
                        for i in range(1, len(path)):
                            if path[i-1] and path[i]:
                                cv2.line(output_frame, path[i-1], path[i], (255,0,0), 2)

                    y_pos = 30
                    for key, value in status.items():
                        cv2.putText(output_frame, f"{key}: {value}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_pos += 20

                    speed_text = f"L: {speeds.get('left', 0):.1f} | R: {speeds.get('right', 0):.1f}"
                    cv2.putText(output_frame, speed_text, (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # A conversão final para RGB deve acontecer aqui, antes de encodar.
                # O browser espera um JPEG no formato RGB.
                final_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                ret, buffer = cv2.imencode('.jpg', final_frame_rgb)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                log(f"Erro no stream de vídeo: {e}")
                # Em caso de erro, gera um frame de erro para não quebrar o stream
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Erro no Stream", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.03)

# --- Handlers de SocketIO ---
@socketio.on('connect')
def handle_connect():
    log("Cliente conectado ao WebSocket")

@socketio.on('command')
def handle_command(data):
    command = data.get('command')
    payload = data.get('payload', {})
    log(f"Comando '{command}' recebido via WebSocket com payload: {payload}")

    if command == 'start_robot':
        SHARED_STATE['start_event'].set()
    elif command == 'set_view_mode':
        with SHARED_STATE['stream_lock']:
            SHARED_STATE['stream_data']['view_mode'] = payload.get('mode', 'normal')
        log(f"Modo de visualização alterado para: {SHARED_STATE['stream_data']['view_mode']}")
    elif command == 'calibrate_pixel':
        SHARED_STATE['calibration_request'] = payload
        log(f"Requisição de calibração recebida: {payload}")
    elif command == 'save_config':
        log("Comando para salvar configuração recebido.")
    else:
        log(f"Comando desconhecido recebido: {command}")

def run_stream():
    socketio.run(app, host='0.0.0.0', port=5000)
