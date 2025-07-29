import cv2
import numpy as np
import os
import json
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# --- Configuração e Sincronização ---
config = {
    "pid": {"kp": 0.4, "ki": 0.0, "kd": 0.1},
    "hsv_black": { "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50]) }
}
start_event = threading.Event()
stream_lock = threading.Lock()

# --- Dados para o Stream e Logs ---
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
last_mask = np.zeros((480, 640), dtype=np.uint8)
path_history = []
motor_speeds = {"left": 0, "right": 0}
status_data = {}
view_mode = 'normal'
logs = []
MAX_LOGS = 20

# --- Funções de Controle ---
def log(message):
    """ Adiciona uma mensagem ao log global. """
    with stream_lock:
        timestamp = time.strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {message}")
        if len(logs) > MAX_LOGS:
            logs.pop(0)

def load_config():
    if os.path.exists('config.json'):
        with stream_lock:
            with open('config.json', 'r') as f:
                loaded_config = json.load(f)
                config['pid'] = loaded_config['pid']
                config['hsv_black']['lower'] = np.array(loaded_config['hsv_black']['lower'])
                config['hsv_black']['upper'] = np.array(loaded_config['hsv_black']['upper'])
                print("Configuração carregada de config.json")
load_config()

def save_config():
    with stream_lock:
        with open('config.json', 'w') as f:
            config_to_save = {
                'pid': config['pid'],
                'hsv_black': {
                    'lower': config['hsv_black']['lower'].tolist(),
                    'upper': config['hsv_black']['upper'].tolist()
                }
            }
            json.dump(config_to_save, f, indent=4)

def update_stream_data(frame, mask, new_path_history, new_motor_speeds, new_status_data):
    with stream_lock:
        global last_frame, last_mask, path_history, motor_speeds, status_data
        last_frame = frame.copy()
        last_mask = mask.copy()
        path_history = list(new_path_history)
        motor_speeds = dict(new_motor_speeds)
        status_data = dict(new_status_data)

# --- Rotas Flask ---
@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return render_template_string(f.read(), config=config)

@app.route('/start', methods=['POST'])
def start_robot():
    start_event.set()
    return jsonify(success=True)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    with stream_lock:
        config['pid']['kp'] = float(data['kp'])
        config['pid']['ki'] = float(data['ki'])
        config['pid']['kd'] = float(data['kd'])
        config['hsv_black']['lower'] = np.array([int(data['h_min_black']), int(data['s_min_black']), int(data['v_min_black'])])
        config['hsv_black']['upper'] = np.array([int(data['h_max_black']), int(data['s_max_black']), int(data['v_max_black'])])
    save_config()
    return jsonify(success=True)

@app.route('/set_view_mode', methods=['POST'])
def set_view_mode_route():
    global view_mode
    with stream_lock:
        view_mode = request.json.get('mode', 'normal')
    return jsonify(success=True)

@app.route('/logs')
def get_logs():
    """ Fornece os logs em formato JSON. """
    with stream_lock:
        return jsonify(logs=list(logs))

@app.route('/stream')
def stream():
    def generate():
        while True:
            with stream_lock:
                output_frame = last_frame.copy()
                current_path = list(path_history)
                current_mask = last_mask.copy()
                current_view = view_mode
                current_status = dict(status_data)
                current_speeds = dict(motor_speeds)

            if current_view == 'mask':
                output_frame = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
            elif current_view == 'contours':
                contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)

            for i in range(1, len(current_path)):
                cv2.line(output_frame, current_path[i - 1], current_path[i], (255, 0, 0), 2)

            if current_view != 'mask':
                y_pos = 30

                # Desenha o status
                if current_status:
                    for key, value in current_status.items():
                        # Formata valores booleanos e floats para melhor visualização
                        if isinstance(value, bool):
                            display_value = "Sim" if value else "Nao"
                            color = (0, 255, 0) if value else (0, 0, 255)
                        elif isinstance(value, float):
                            display_value = f"{value:.2f}"
                            color = (0, 255, 255)
                        else:
                            display_value = str(value)
                            color = (0, 255, 255)

                        text = f"{key}: {display_value}"
                        cv2.putText(output_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_pos += 20

                # Desenha a velocidade
                speed_text = f"L: {current_speeds.get('left', 0):.1f} | R: {current_speeds.get('right', 0):.1f}"
                cv2.putText(output_frame, speed_text, (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rotated_frame = cv2.rotate(output_frame, cv2.ROTATE_90_CLOCKWISE)
            ret, buffer = cv2.imencode('.jpg', rotated_frame)
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    app.run(host='0.0.0.0', port=5000)
