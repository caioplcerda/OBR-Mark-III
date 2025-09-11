import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple

# ====== CONFIGURAÇÕES ======
USE_LEDS = True
WEB_AVAILABLE = True
GPIO_AVAILABLE = True

try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except Exception:
    GPIO_AVAILABLE = False

# Pinos de sensores de obstáculo
SENSOR_FRONT = 17
SENSOR_LEFT = 27
SENSOR_RIGHT = 22

if GPIO_AVAILABLE:
    GPIO.setup(SENSOR_FRONT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SENSOR_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SENSOR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

CONFIG_DEFAULTS = {
    "camera": {"width": 640, "height": 480, "rotate_180": False},
    "robot": {
        "max_speed": 100,
        "base_speed": 50,  # velocidade base mais segura
        "k_gain": 0.025,
        "k_derivative": 0.015,
        "dead_zone": 8
    }
}


class LineFollowerController:
    """Controlador PD para seguir linha com busca quando perder o traço."""
    def __init__(self, config: Dict):
        self.max_speed = config["robot"]["max_speed"]
        self.base_speed = config["robot"]["base_speed"]
        self.k_gain = config["robot"]["k_gain"]
        self.k_derivative = config["robot"]["k_derivative"]
        self.dead_zone = config["robot"]["dead_zone"]
        self.error_history = deque(maxlen=5)
        self.last_error = None
        self.lost_counter = 0

    def compute_speeds(self, centroid: Optional[Tuple[int, int]], width: int) -> Tuple[int, int]:
        """Calcula velocidade com PD ou faz busca caso perca a linha."""
        if centroid is None:
            self.lost_counter += 1
            # Gira lentamente para procurar linha
            if self.lost_counter < 15:
                return 30, -30  # gira para a esquerda
            else:
                return -30, 30  # gira para a direita após algumas tentativas

        self.lost_counter = 0  # reset se encontrou linha

        # Erro do centro
        error = centroid[0] - (width // 2)
        self.error_history.append(error)
        avg_error = sum(self.error_history) / len(self.error_history)

        derivative = (error - self.last_error) if self.last_error is not None else 0
        self.last_error = error

        # Correção PD
        correction = int(self.k_gain * avg_error + self.k_derivative * derivative)

        # Velocidade adaptativa: reduz em curvas
        adaptive_base_speed = max(30, self.base_speed - int(abs(error) * 0.05))

        left = adaptive_base_speed - correction
        right = adaptive_base_speed + correction

        # Zona morta (quando quase centralizado)
        if abs(avg_error) < self.dead_zone:
            left = right = adaptive_base_speed

        return (
            int(np.clip(left, -self.max_speed, self.max_speed)),
            int(np.clip(right, -self.max_speed, self.max_speed))
        )


class Robot:
    def __init__(self):
        self.config = CONFIG_DEFAULTS
        self.width = self.config["camera"]["width"]
        self.height = self.config["camera"]["height"]

        from hardware_control import HardwareControl
        from vision import Vision

        self.vision = Vision({}, print)
        self.hardware = HardwareControl({"pid": {}})
        self.controller = LineFollowerController(self.config)

        self.running = False
        self.thread = None
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_sensors(self) -> bool:
        """Retorna True se houver obstáculo na frente."""
        if not GPIO_AVAILABLE:
            return False
        return GPIO.input(SENSOR_FRONT) == GPIO.LOW or GPIO.input(SENSOR_LEFT) == GPIO.LOW or GPIO.input(SENSOR_RIGHT) == GPIO.LOW

    def _binarize(self, frame: np.ndarray) -> np.ndarray:
        """Binariza imagem com ROI para reduzir ruídos."""
        # Recorta só a parte inferior
        roi = frame[self.height - 120:self.height, 0:self.width]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        return bw

    def start(self):
        if self.running:
            print("Robô já está em execução.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.hardware.stop()
        if self.cap.isOpened():
            self.cap.release()
        print("Robô parado.")

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Se detectar obstáculo, parar imediatamente
            if self.read_sensors():
                self.hardware.stop()
                print("Obstáculo detectado! Parando robô.")
                time.sleep(0.5)
                continue

            bw = self._binarize(frame)
            centroid, _ = self.vision.strip_centroid(bw, self.height - 50, 40)
            left_speed, right_speed = self.controller.compute_speeds(centroid, self.width)

            self.hardware.drive(left_speed, right_speed)
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        robot = Robot()
        robot.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        robot.stop()
