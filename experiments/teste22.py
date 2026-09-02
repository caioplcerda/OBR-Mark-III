#!/usr/bin/env python3
# line_follower.py
# Seguidor de linha usando Picamera3, TB6612FNG e 21 NeoPixels em GPIO12
# Pinos dos motores (BCM) conforme informado pelo usuário:
# Esquerdo: BIN1=17, BIN2=27, PWMB=22
# Direito:  BIN1=23, BIN2=24, PWMB=25
# NeoPixels: GPIO12, 21 LEDs

import time
import logging
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera3 import Picamera3
try:
    import board
    import neopixel
    NEOPIXEL_AVAILABLE = True
except Exception as e:
    NEOPIXEL_AVAILABLE = False
    print("Warning: neopixel library not available. LEDs will be ignorados.", e)

# -------- CONFIGURAÇÃO HARDWARE --------
# Motor left
L_BIN1 = 17
L_BIN2 = 27
L_PWMB = 22

# Motor right
R_BIN1 = 23
R_BIN2 = 24
R_PWMB = 25

# NeoPixels
NEO_PIN = board.D12 if NEOPIXEL_AVAILABLE else None
NEO_COUNT = 21

# -------- CONFIGURAÇÃO DE CONTROLE --------
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CROP_TOP = 220         # cortar a parte superior, focar na região próxima ao robô
CROP_BOTTOM = 320
FPS = 30

BASE_SPEED = 40        # velocidade base (0..100)
MAX_SPEED = 85         # limite máximo de PWM duty cycle
MIN_SPEED = 20         # mínimo prático para mover motores

# PID para direção
Kp = 0.9
Ki = 0.0
Kd = 0.08
INTEGRAL_LIMIT = 100

# Thresholds e parâmetros de visão
GAUSSIAN_BLUR = (5,5)
THRESH_BLOCK_SIZE = 31
THRESH_C = 10
MORPH_KERNEL = (5,5)
LOST_LINE_PIXELS_THRESHOLD = 200  # se menos que isso, consideramos perdido
INTERSECTION_PIXELS_THRESHOLD = 3000 # se mais que isso -> possível cruzamento/interseção
CURVA_FORTE_ERROR_RATIO = 0.5  # erro normalizado (0..1) acima disso = curva 90 provável

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# -------- INICIALIZAÇÃO GPIO --------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [L_BIN1, L_BIN2, L_PWMB, R_BIN1, R_BIN2, R_PWMB]
for p in motor_pins:
    GPIO.setup(p, GPIO.OUT)

# Configura PWMs (frequência 1000Hz)
freq = 1000
pwm_L = GPIO.PWM(L_PWMB, freq)
pwm_R = GPIO.PWM(R_PWMB, freq)
pwm_L.start(0)
pwm_R.start(0)

# -------- INITIALIZAÇÃO NEOPIXELS --------
if NEOPIXEL_AVAILABLE:
    pixels = neopixel.NeoPixel(NEO_PIN, NEO_COUNT, brightness=0.4, auto_write=True)
    def set_pixels_color(color):
        pixels.fill(color)
else:
    def set_pixels_color(color):
        pass

# Cores helper
def rgb_tuple(hex_color):
    # hex_color like (r,g,b) already -> return
    return tuple(hex_color)

RED   = rgb_tuple((255, 0, 0))
GREEN = rgb_tuple((0, 255, 0))
YELLOW= rgb_tuple((255, 150, 0))
OFF   = rgb_tuple((0,0,0))

# -------- FUNÇÕES DE MOTOR --------
def clamp(v, a, b):
    return max(a, min(b, v))

def set_motor_speed(left_speed, right_speed):
    """
    left_speed and right_speed in range [-100..100]
    Positive -> forward, Negative -> backward
    """
    # Left
    if left_speed >= 0:
        GPIO.output(L_BIN1, GPIO.HIGH)
        GPIO.output(L_BIN2, GPIO.LOW)
    else:
        GPIO.output(L_BIN1, GPIO.LOW)
        GPIO.output(L_BIN2, GPIO.HIGH)
    # Right
    if right_speed >= 0:
        GPIO.output(R_BIN1, GPIO.HIGH)
        GPIO.output(R_BIN2, GPIO.LOW)
    else:
        GPIO.output(R_BIN1, GPIO.LOW)
        GPIO.output(R_BIN2, GPIO.HIGH)

    # Duty cycle (abs)
    dL = clamp(abs(left_speed), 0, 100)
    dR = clamp(abs(right_speed), 0, 100)
    pwm_L.ChangeDutyCycle(dL)
    pwm_R.ChangeDutyCycle(dR)

def stop_motors():
    pwm_L.ChangeDutyCycle(0)
    pwm_R.ChangeDutyCycle(0)
    GPIO.output(L_BIN1, GPIO.LOW)
    GPIO.output(L_BIN2, GPIO.LOW)
    GPIO.output(R_BIN1, GPIO.LOW)
    GPIO.output(R_BIN2, GPIO.LOW)

# -------- FUNÇÕES DE VISÃO --------
def preprocess(frame):
    # frame: BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped = gray[CROP_TOP:CROP_BOTTOM, :]
    blurred = cv2.GaussianBlur(cropped, GAUSSIAN_BLUR, 0)
    # Usamos threshold adaptativo porque condições de iluminação podem variar
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, THRESH_BLOCK_SIZE, THRESH_C)
    # morfologia para limpar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    morphed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return morphed

def find_line_center(binary_img):
    """
    Retorna (cx, area_pixels, lost_flag)
    cx: coordenada x do centro da linha (0..width-1) na imagem recortada
    """
    h, w = binary_img.shape
    # Soma por coluna para priorizar a linha (proporciona robustez)
    col_sum = np.sum(binary_img, axis=0)  # quantidade de "preto" (255) por coluna
    # Convert 255 escala -> counts:
    col_sum = col_sum / 255.0
    total_pixels = np.sum(col_sum)
    if total_pixels < LOST_LINE_PIXELS_THRESHOLD:
        return None, int(total_pixels), True

    # Centroid calculado por média ponderada das colunas
    indices = np.arange(w)
    cx = int(np.sum(indices * col_sum) / (np.sum(col_sum) + 1e-6))
    return cx, int(total_pixels), False

# -------- PID CONTROLLER --------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.prev_error = 0.0
        self.sum_error = 0.0
        self.last_time = None

    def reset(self):
        self.prev_error = 0.0
        self.sum_error = 0.0
        self.last_time = None

    def update(self, error):
        now = time.time()
        dt = 0.0
        if self.last_time is None:
            dt = 0.0
        else:
            dt = now - self.last_time
        self.last_time = now

        self.sum_error += error * (dt if dt>0 else 1.0)
        # anti-windup
        self.sum_error = clamp(self.sum_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)

        d_error = (error - self.prev_error) / (dt if dt>0 else 1.0)
        self.prev_error = error

        out = self.kp * error + self.ki * self.sum_error + self.kd * d_error
        return out

pid = PID(Kp, Ki, Kd)

# -------- CAMERA --------
picam = Picamera3()
config = picam.create_still_configuration({"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam.configure(config)
time.sleep(0.5)
picam.start()

# -------- LOOP PRINCIPAL --------
def main_loop():
    logging.info("Iniciando line follower")
    set_pixels_color(GREEN)
    pid.reset()
    try:
        while True:
            start = time.time()
            frame = picam.capture_array()
            binary = preprocess(frame)
            cx_area, area_pixels, lost = find_line_center(binary)

            h, w = binary.shape
            center_x = w // 2

            if lost:
                # Linha perdida
                logging.warning(f"Linha perdida (pixels detectados: {area_pixels}). Parando e procurando...")
                set_pixels_color(RED)
                # estratégia simples: girar para tentar encontrar: girar levemente à esquerda por um instante
                set_motor_speed(-BASE_SPEED//2, BASE_SPEED//2)  # girar no próprio eixo
                time.sleep(0.12)
                stop_motors()
                pid.reset()
                continue
            else:
                set_pixels_color(GREEN)

            # Erro normalizado: negative = esquerda (cx < center), positive = direita
            error_pixels = (cx_area - center_x)
            error_norm = error_pixels / float(center_x)  # -1 .. 1

            # Detectar interseção (muitos pixels escuros)
            if area_pixels > INTERSECTION_PIXELS_THRESHOLD:
                logging.info(f"Possível interseção detectada (area_pixels={area_pixels}).")
                # sinaliza amarelo por um curto período
                set_pixels_color(YELLOW)
                # pode-se implementar lógica de interseção aqui; por ora reduz velocidade e segue
                time.sleep(0.05)
                set_pixels_color(GREEN)

            # Detectar curva forte (quando erro normalizado grande)
            if abs(error_norm) > CURVA_FORTE_ERROR_RATIO:
                logging.info(f"Curva forte detectada. erro_norm={error_norm:.2f}")
                # sinaliza amarelo curto
                set_pixels_color(YELLOW)
            else:
                set_pixels_color(GREEN)

            # PID output é a correção (positivo -> virar pra direita)
            correction = pid.update(error_norm)  # pode ser >1 ou < -1 dependendo ganho
            # Mapear correção para ajuste de velocidade diferencial
            # compute left and right speeds
            # quando correction positivo, reduz velocidade do lado direito e aumenta do esquerdo para virar direita
            turn_scale = clamp(correction * 80.0, -100, 100)  # escala arbitrária para convergir em PWM
            left = BASE_SPEED + turn_scale
            right = BASE_SPEED - turn_scale

            # Limites
            left = clamp(left, -MAX_SPEED, MAX_SPEED)
            right = clamp(right, -MAX_SPEED, MAX_SPEED)

            # Evitar valores muito baixos que não movem o motor
            if 0 < abs(left) < MIN_SPEED:
                left = MIN_SPEED * np.sign(left)
            if 0 < abs(right) < MIN_SPEED:
                right = MIN_SPEED * np.sign(right)

            set_motor_speed(left, right)

            # debug log
            logging.debug(f"cx={cx_area} center={center_x} err={error_norm:.3f} corr={correction:.3f} L={left:.1f} R={right:.1f}")

            # manter loop em ~FPS
            elapsed = time.time() - start
            to_sleep = max(0.0, (1.0 / FPS) - elapsed)
            time.sleep(to_sleep)

    except KeyboardInterrupt:
        logging.info("Interrompido pelo usuário. Parando motores.")
    finally:
        set_pixels_color(OFF)
        stop_motors()
        picam.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()
