from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)
last_frame = np.zeros((240, 640, 3), dtype=np.uint8)
path_history = []
LOOKAHEAD_STEPS = 5

def update_frame(frame, new_path_history):
    """ Atualiza o frame e o histórico do caminho para o stream. """
    global last_frame, path_history
    last_frame = frame
    path_history = new_path_history

@app.route('/stream')
def stream():
    """ Gera o stream de vídeo com overlay. """
    def generate():
        global last_frame
        while True:
            overlay = last_frame.copy()

            # Desenha a projeção look-ahead
            if len(path_history) >= 2:
                pts = np.float32(path_history[-LOOKAHEAD_STEPS:])
                ys = pts[:, 1]
                xs = pts[:, 0]
                A = np.vstack([ys, np.ones_like(ys)]).T
                a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
                for y in range(160, 240, 5):
                    x = int(a * y + b)
                    cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)

            # Desenha o histórico do caminho
            for i in range(1, len(path_history)):
                cv2.line(overlay, path_history[i - 1], path_history[i], (255, 0, 0), 2)
                cv2.circle(overlay, path_history[i], 3, (0, 255, 255), -1)

            # Rotaciona a imagem para visualização em pé
            rotated = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

            ret, buffer = cv2.imencode('.jpg', rotated)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    """ Executa o servidor Flask em uma thread separada. """
    app.run(host='0.0.0.0', port=5000)
