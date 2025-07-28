from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# Objeto de configuração para compartilhar os parâmetros
config = {
    "pid": {"kp": 0.4, "ki": 0.0, "kd": 0.1},
    "hsv_black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 50])
    }
}

# --- Rotas da Interface Web ---

@app.route('/')
def index():
    """ Serve a página de calibração. """
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """ Recebe e aplica os novos parâmetros de calibração. """
    data = request.json

    # Atualiza PID
    config['pid']['kp'] = float(data['kp'])
    config['pid']['ki'] = float(data['ki'])
    config['pid']['kd'] = float(data['kd'])

    # Atualiza HSV Preto
    config['hsv_black']['lower'] = np.array([int(data['h_min_black']), int(data['s_min_black']), int(data['v_min_black'])])
    config['hsv_black']['upper'] = np.array([int(data['h_max_black']), int(data['s_max_black']), int(data['v_max_black'])])

    print(f"Novos parâmetros recebidos: {config}")
    return jsonify(success=True)

# --- Lógica do Stream de Vídeo ---

last_frame = np.zeros((240, 640, 3), dtype=np.uint8)
path_history = []
LOOKAHEAD_STEPS = 5

def update_frame(frame, new_path_history):
    """ Atualiza o frame para o stream. """
    global last_frame, path_history
    last_frame = frame
    path_history = new_path_history

@app.route('/stream')
def stream():
    """ Gera o stream de vídeo. """
    def generate():
        global last_frame
        while True:
            # ... (lógica de overlay permanece a mesma) ...
            ret, buffer = cv2.imencode('.jpg', last_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    """ Executa o servidor Flask. """
    app.run(host='0.0.0.0', port=5000)
