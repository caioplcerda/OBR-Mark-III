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
        "hsv_black": { "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50]) },
        "hsv_white": { "lower": np.array([0, 0, 180]), "upper": np.array([180, 25, 255]) }
    },
    "start_event": threading.Event(),
    "stream_lock": threading.Lock(),
    "stream_data": {
        "last_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "last_mask": np.zeros((480, 640), dtype=np.uint8),
        "path_history": [],
        "motor_speeds": {"left": 0, "right": 0},
        "status_data": {},
        "view_mode": 'normal',
        "derivative_scan": None
    },
    "calibration_request": None
}

# --- Funções de Logging ---
def log(message):
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)  # Mantém o log no console do servidor
    socketio.emit('log_message', {'data': full_message})

def create_derivative_graph(derivative_data, width, height):
    """ Gera uma imagem com o gráfico da derivada da varredura. """
    if derivative_data is None or len(derivative_data) == 0:
        graph = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
        cv2.putText(graph, "No derivative data", (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        return graph

    graph = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    center_y = height // 2
    cv2.line(graph, (0, center_y), (width, center_y), (150, 150, 150), 1)

    num_points = len(derivative_data)
    x_coords = np.linspace(0, width - 1, num_points).astype(int)

    max_val = np.max(np.abs(derivative_data))
    if max_val < 1: max_val = 1.0

    y_coords = center_y - (derivative_data * (height / 2 - 10) / max_val).astype(int)

    points = np.column_stack((x_coords, y_coords))
    cv2.polylines(graph, [points], isClosed=False, color=(0, 0, 255), thickness=1)

    left_edge_idx = np.argmax(derivative_data)
    right_edge_idx = np.argmin(derivative_data)

    cv2.circle(graph, (x_coords[left_edge_idx], y_coords[left_edge_idx]), 7, (0, 255, 0), -1)
    cv2.circle(graph, (x_coords[right_edge_idx], y_coords[right_edge_idx]), 7, (255, 0, 0), -1)

    return graph


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
                    derivative_scan = s_data.get('derivative_scan')

                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                if view_mode == 'mask':
                    output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                elif view_mode == 'derivative':
                    output_frame = create_derivative_graph(derivative_scan, frame.shape[1], frame.shape[0])
                else:
                    output_frame = frame
                    if view_mode == 'contours' and mask is not None and mask.size > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)

                    if path:
                        # Desenha as linhas de conexão (azul)
                        for i in range(1, len(path)):
                            if path[i-1] and path[i]:
                                p1 = tuple(map(int, path[i-1]))
                                p2 = tuple(map(int, path[i]))
                                cv2.line(output_frame, p1, p2, (255, 0, 0), 2)
                        # Desenha os pontos detectados (vermelho)
                        for point in path:
                            p = tuple(map(int, point))
                            cv2.circle(output_frame, p, 5, (0, 0, 255), -1)


                    y_pos = 30
                    for key, value in status.items():
                        text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                        cv2.putText(output_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_pos += 20

                    speed_text = f"L: {speeds.get('left', 0):.1f} | R: {speeds.get('right', 0):.1f}"
                    cv2.putText(output_frame, speed_text, (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Converte o frame para RGB antes de encodar para JPEG, corrigindo as cores no navegador.
                output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                ret, buffer = cv2.imencode('.jpg', output_frame_rgb)
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

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
