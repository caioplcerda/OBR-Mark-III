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
motor_speeds = {"left": 0, "right": 0}
status_data = {}
LOOKAHEAD_STEPS = 5

def update_stream_data(frame, new_path_history, new_motor_speeds, new_status_data):
    """ Atualiza todos os dados para o stream. """
    global last_frame, path_history, motor_speeds, status_data
    # Garante que estamos passando uma cópia para evitar race conditions
    last_frame = frame.copy()
    path_history = new_path_history
    motor_speeds = new_motor_speeds
    status_data = new_status_data

@app.route('/stream')
def stream():
    """ Gera o stream de vídeo. """
    def generate():
        global last_frame, motor_speeds, status_data
        while True:
            overlay = last_frame.copy()

            # Adiciona informações de status ao overlay
            y_pos = 30
            if status_data:
                for key, value in status_data.items():
                    text = f"{key}: {value}"
                    cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_pos += 20

            # Adiciona o texto de velocidade dos motores
            speed_text = f"L: {motor_speeds.get('left', 0):.1f} | R: {motor_speeds.get('right', 0):.1f}"
            cv2.putText(overlay, speed_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Rotaciona o frame final para exibição
            rotated_frame = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

            ret, buffer = cv2.imencode('.jpg', rotated_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    """ Executa o servidor Flask. """
    app.run(host='0.0.0.0', port=5000)
