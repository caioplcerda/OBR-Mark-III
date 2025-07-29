from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import os
import json

app = Flask(__name__)

# Objeto de configuração e comando de início
config = {
    "pid": {"kp": 0.4, "ki": 0.0, "kd": 0.1},
    "hsv_black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 50])
    }
}
start_command_received = False

# --- Rotas da Interface Web ---

@app.route('/start', methods=['POST'])
def start_robot():
    """ Recebe o comando de início da web. """
    global start_command_received
    start_command_received = True
    print("Comando de início recebido da web.")
    return jsonify(success=True)

@app.route('/')
def index():
    """ Serve a página de calibração. """
    with open('index.html', 'r') as f:
        return render_template_string(f.read(), config=config)

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

    # Salva a configuração em um arquivo JSON
    save_config()

    print(f"Novos parâmetros recebidos e salvos: {config}")
    return jsonify(success=True)

def save_config():
    """ Salva o objeto de configuração em config.json. """
    with open('config.json', 'w') as f:
        config_to_save = {
            'pid': config['pid'],
            'hsv_black': {
                'lower': config['hsv_black']['lower'].tolist(),
                'upper': config['hsv_black']['upper'].tolist()
            }
        }
        json.dump(config_to_save, f, indent=4)

def load_config():
    """ Carrega a configuração de config.json, se existir. """
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            loaded_config = json.load(f)
            config['pid'] = loaded_config['pid']
            config['hsv_black']['lower'] = np.array(loaded_config['hsv_black']['lower'])
            config['hsv_black']['upper'] = np.array(loaded_config['hsv_black']['upper'])
            print("Configuração carregada de config.json")

# Carrega a configuração ao iniciar
load_config()

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
