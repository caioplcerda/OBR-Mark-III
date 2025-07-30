import cv2
import numpy as np
import os
import json
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# --- Dicionário de Estado Global Compartilhado ---
SHARED_STATE = {
    "config": {
        "pid": {"kp": 0.4, "ki": 0.0, "kd": 0.1},
        "hsv_black": { "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50]) }
    },
    "start_event": threading.Event(),
    "stream_lock": threading.Lock(),
    "logs": [],
    "MAX_LOGS": 20,
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

# --- Funções de Controle e Logging ---
def log(message):
    with SHARED_STATE["stream_lock"]:
        timestamp = time.strftime("%H:%M:%S")
        SHARED_STATE["logs"].append(f"[{timestamp}] {message}")
        if len(SHARED_STATE["logs"]) > SHARED_STATE["MAX_LOGS"]:
            SHARED_STATE["logs"].pop(0)

# --- Rotas Flask ---
@app.route('/')
def index():
    return render_template_string(open('index.html').read(), config=SHARED_STATE['config'])

@app.route('/start', methods=['POST'])
def start_robot_route():
    SHARED_STATE['start_event'].set()
    log("Comando de início recebido pela web.")
    return jsonify(success=True)

@app.route('/calibrate', methods=['POST'])
def calibrate_route():
    data = request.json
    with SHARED_STATE['stream_lock']:
        SHARED_STATE['config']['pid']['kp'] = float(data['kp'])
        SHARED_STATE['config']['pid']['ki'] = float(data['ki'])
        SHARED_STATE['config']['pid']['kd'] = float(data['kd'])
        SHARED_STATE['config']['hsv_black']['lower'] = np.array([int(data['h_min_black']), int(data['s_min_black']), int(data['v_min_black'])])
        SHARED_STATE['config']['hsv_black']['upper'] = np.array([int(data['h_max_black']), int(data['s_max_black']), int(data['v_max_black'])])
    log("Configuração PID/HSV atualizada.")
    return jsonify(success=True)

@app.route('/set_view_mode', methods=['POST'])
def set_view_mode_route():
    with SHARED_STATE['stream_lock']:
        SHARED_STATE['stream_data']['view_mode'] = request.json.get('mode', 'normal')
    log(f"Modo de visualização alterado para: {SHARED_STATE['stream_data']['view_mode']}")
    return jsonify(success=True)

@app.route('/calibrate_pixel', methods=['POST'])
def calibrate_pixel_route():
    data = request.json
    x, y, color_name = data['x'], data['y'], data['color']

    # Acessa o último frame e chama a função de calibração
    with SHARED_STATE['stream_lock']:
        frame = SHARED_STATE['stream_data']['last_frame']

    # A chamada à visão precisa ser feita na thread principal
    # Vamos usar um evento para sinalizar a calibração
    SHARED_STATE['calibration_request'] = {'x': x, 'y': y, 'color': color_name}

    return jsonify(success=True)

@app.route('/logs')
def get_logs_route():
    with SHARED_STATE['stream_lock']:
        return jsonify(logs=list(SHARED_STATE['logs']))

@app.route('/stream')
def stream_route():
    def generate():
        while True:
            with SHARED_STATE['stream_lock']:
                s_data = SHARED_STATE['stream_data']
                frame = s_data['last_frame'].copy()
                mask = s_data['last_mask'].copy()
                view_mode = s_data['view_mode']
                path = list(s_data['path_history'])
                status = dict(s_data['status_data'])
                speeds = dict(s_data['motor_speeds'])

            if view_mode == 'mask':
                output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                output_frame = frame
                if view_mode == 'contours':
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)

                for i in range(1, len(path)):
                    cv2.line(output_frame, path[i-1], path[i], (255,0,0), 2)

                y_pos = 30
                for key, value in status.items():
                    cv2.putText(output_frame, f"{key}: {value}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_pos += 20

                speed_text = f"L: {speeds.get('left', 0):.1f} | R: {speeds.get('right', 0):.1f}"
                cv2.putText(output_frame, speed_text, (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', output_frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    app.run(host='0.0.0.0', port=5000, threaded=True)
