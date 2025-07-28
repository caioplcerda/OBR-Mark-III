import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import os

# === GPIO SETUP ===
LEFT_MOTOR_FORWARD = 17
LEFT_MOTOR_BACKWARD = 27
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 23
FAN_GPIO = 24  # GPIO para ventoinha externa, se necessário

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(FAN_GPIO, GPIO.OUT)
GPIO.output(FAN_GPIO, GPIO.HIGH)  # Liga ventoinha externa no GPIO 24

# === Liga ventoinha embutida do Pi5 se disponível ===
try:
    if os.path.exists("/usr/bin/rpi-fancontrol"):
        os.system("sudo rpi-fancontrol --fan 1")
    elif os.path.exists("/proc/device-tree/thermal-zones/fan-thermal/cooling-device"):
        os.system("echo 1 | sudo tee /sys/class/thermal/cooling_device0/cur_state")
    else:
        print("[INFO] Ventoinha embutida não detectada por script")
except Exception as e:
    print(f"[WARNING] Erro ao tentar ativar ventoinha embutida: {e}")

# PWM setup (opcional)
left_pwm_fwd = GPIO.PWM(LEFT_MOTOR_FORWARD, 100)
right_pwm_fwd = GPIO.PWM(RIGHT_MOTOR_FORWARD, 100)
left_pwm_fwd.start(0)
right_pwm_fwd.start(0)

# === Inicializa câmera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 240)}))
picam2.start()
time.sleep(1)

# === Limiar HSV ===
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 50])
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])

# === Parâmetros ===
FRAME_WIDTH = 640
CENTER_X = FRAME_WIDTH // 2
BASE_SPEED = 50
KP = 0.4
GREEN_THRESHOLD_AREA = 5000
PATH_HISTORY_LENGTH = 20
LOOKAHEAD_STEPS = 5

# === Parâmetros de desvio de obstáculo ===
OBSTACLE_MIN_AREA = 2000
OBSTACLE_REGION_Y = 100  # linha horizontal onde procuramos obstáculo
OBSTACLE_AVOID_SPEED = 40
OBSTACLE_AVOID_TIME = 0.5  # ajuste este valor para controlar o tempo de desvio lateral

# === Controle de motores ===
def set_motor_speed(left, right):
    left = max(0, min(100, left))
    right = max(0, min(100, right))
    left_pwm_fwd.ChangeDutyCycle(left)
    right_pwm_fwd.ChangeDutyCycle(right)

def stop():
    set_motor_speed(0, 0)

# === Detecção de linha e verde ===
def detect_features(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    roi_line = mask_black[160:240, :]
    M = cv2.moments(roi_line)
    cx = -1
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])

    green_detected = cv2.countNonZero(mask_green) > GREEN_THRESHOLD_AREA

    # ROI para obstáculo: verifica interrupção brusca na linha preta no centro
    obstacle = False
    obstacle_roi = roi_line[OBSTACLE_REGION_Y-10:OBSTACLE_REGION_Y+10, CENTER_X-40:CENTER_X+40]
    contours, _ = cv2.findContours(obstacle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > OBSTACLE_MIN_AREA:
            obstacle = True
            break

    return cx, green_detected, obstacle, mask_black, mask_green

# === Flask app para stream ===
app = Flask(__name__)
last_frame = np.zeros((240, 640, 3), dtype=np.uint8)
path_history = []

@app.route('/stream')
def stream():
    def generate():
        global last_frame
        while True:
            overlay = last_frame.copy()
            if len(path_history) >= 2:
                pts = np.float32(path_history[-LOOKAHEAD_STEPS:])
                ys = pts[:, 1]
                xs = pts[:, 0]
                A = np.vstack([ys, np.ones_like(ys)]).T
                a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
                for y in range(160, 240, 5):
                    x = int(a * y + b)
                    cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)
            rotated = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
            ret, buffer = cv2.imencode('.jpg', rotated)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Thread principal ===
def main_loop():
    global last_frame, path_history
    try:
        while True:
            frame = picam2.capture_array()
            cx, green, obstacle, mask_black, mask_green = detect_features(frame)

            if obstacle:
                status = "Desviando de obstáculo."
                # Estratégia de desvio simples: gira para esquerda temporariamente
                set_motor_speed(OBSTACLE_AVOID_SPEED, 0)
                time.sleep(OBSTACLE_AVOID_TIME)
                set_motor_speed(BASE_SPEED, BASE_SPEED)
                time.sleep(0.3)  # tempo para retomar linha após desvio
                continue

            elif cx != -1:
                error = CENTER_X - cx
                correction = KP * error
                left_speed = BASE_SPEED + correction
                right_speed = BASE_SPEED - correction
                path_history.append((cx, 200))
                if len(path_history) > PATH_HISTORY_LENGTH:
                    path_history.pop(0)
                set_motor_speed(left_speed, right_speed)
                status = f"L: {int(left_speed)} R: {int(right_speed)}"
            else:
                status = "Linha perdida. Parando."
                stop()

            for i in range(1, len(path_history)):
                cv2.line(frame, path_history[i - 1], path_history[i], (255, 0, 0), 2)
                cv2.circle(frame, path_history[i], 3, (0, 255, 255), -1)

            if len(path_history) >= 2:
                pts = np.float32(path_history[-LOOKAHEAD_STEPS:])
                ys = pts[:, 1]
                xs = pts[:, 0]
                A = np.vstack([ys, np.ones_like(ys)]).T
                a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
                for y in range(160, 240, 5):
                    x = int(a * y + b)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            last_frame = frame.copy()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[EXIT] Encerrando...")
        stop()
        GPIO.cleanup()

# === Inicialização ===
if __name__ == '__main__':
    t = threading.Thread(target=main_loop)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000)
