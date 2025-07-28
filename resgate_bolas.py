# resgate_bolas.py - Lógica de resgate de bolas para OBR 2025 com varredura (radar)

import time
import cv2
import numpy as np
import RPi.GPIO as GPIO

# === GPIO SETUP (VOCÊ DEFINE ESTES DEPOIS) ===
SERVO_GARRA_1 = None  # abrir/fechar garra
SERVO_GARRA_2 = None  # levantar/abaixar garra
SERVO_RESERVATORIO = None  # libera bolas prateadas traseiras
MOTOR_ESQUERDO = None
MOTOR_DIREITO = None

# === PWM Setup (configure quando tiver os pinos) ===
GPIO.setmode(GPIO.BCM)
servos = []
for pin in [SERVO_GARRA_1, SERVO_GARRA_2, SERVO_RESERVATORIO]:
    if pin is not None:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)
        pwm.start(0)
        servos.append(pwm)
    else:
        servos.append(None)

def set_servo_angle(pwm, angle):
    if pwm:
        duty = 2 + (angle / 18)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)

# === CORES ===
LOWER_SILVER = np.array([0, 0, 180])
UPPER_SILVER = np.array([180, 50, 255])
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 50])

# === Ações com a garra ===
def abrir_garra():
    set_servo_angle(servos[0], 90)

def fechar_garra():
    set_servo_angle(servos[0], 10)

def abaixar_garra():
    set_servo_angle(servos[1], 90)

def levantar_garra():
    set_servo_angle(servos[1], 10)

def soltar_bolas_reservatorio():
    set_servo_angle(servos[2], 90)

def detectar_bolas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_silver = cv2.inRange(hsv, LOWER_SILVER, UPPER_SILVER)
    mask_black = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)

    contornos_silver, _ = cv2.findContours(mask_silver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolas = []
    for c in contornos_silver:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            bolas.append({"tipo": "prata", "pos": (x + w // 2, y + h // 2)})

    for c in contornos_black:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            bolas.append({"tipo": "preta", "pos": (x + w // 2, y + h // 2)})

    return bolas

def pegar_bola(pos):
    print(f"Indo para bola em {pos}")
    abaixar_garra()
    abrir_garra()
    time.sleep(0.3)
    fechar_garra()
    levantar_garra()

def guardar_em_reservatorio():
    print("Guardando bola no reservatório")
    abaixar_garra()
    abrir_garra()
    time.sleep(0.3)
    levantar_garra()
    fechar_garra()

def manter_na_garra():
    print("Mantendo bola na garra")

def ir_para_area(cor):
    print(f"Navegando para área {cor}")
    time.sleep(2)

def varrer_e_detectar(picam2, duracao=6):
    print("Iniciando varredura")
    tempo_inicial = time.time()
    todos_frames = []
    while time.time() - tempo_inicial < duracao:
        frame = picam2.capture_array()
        todos_frames.append(frame.copy())
        # simula giro em lugar com leve delay
        time.sleep(0.4)
    print("Varredura concluída")
    return todos_frames

def executar_resgate(picam2):
    frames_varridos = varrer_e_detectar(picam2)
    bolas_detectadas = []
    for frame in frames_varridos:
        bolas = detectar_bolas(frame)
        bolas_detectadas.extend(bolas)

    prateadas = [b for b in bolas_detectadas if b['tipo'] == 'prata']
    pretas = [b for b in bolas_detectadas if b['tipo'] == 'preta']

    for b in prateadas[:2]:
        pegar_bola(b['pos'])
        guardar_em_reservatorio()

    if pretas:
        pegar_bola(pretas[0]['pos'])
        manter_na_garra()

    ir_para_area("verde")
    soltar_bolas_reservatorio()

    ir_para_area("vermelha")
    abrir_garra()
    abaixar_garra()
    time.sleep(0.5)
    fechar_garra()
    levantar_garra()

# Para integração: chame executar_resgate(picam2) com sua instância de câmera
