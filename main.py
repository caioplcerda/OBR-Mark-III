import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime

# =====================================================
# CONFIGURAÇÕES DO SISTEMA
# =====================================================
USE_GPIO_MOCK = False  # Ativado automaticamente se não estiver na Raspberry Pi
CAM_WIDTH = 640
CAM_HEIGHT = 480

CONFIG = {
    "robot": {
        "max_speed": 100,
        "base_speed": 60,
        "k_gain": 0.05,          # Ganho proporcional
        "k_derivative": 0.02,    # Ganho derivativo
        "dead_zone": 15          # Zona morta para evitar oscilação
    }
}

# =====================================================
# MOCK DO GPIO PARA TESTAR NO PC
# =====================================================
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except (ImportError, RuntimeError):
    USE_GPIO_MOCK = True
    print("[AVISO] GPIO real não encontrado. Usando MOCK GPIO para testes no PC.")

    class MockGPIO:
        BCM = "BCM"
        IN = "IN"
        OUT = "OUT"
        LOW = 0
        HIGH = 1
        PUD_UP = "PUD_UP"

        def setmode(self, *args, **kwargs): pass
        def setwarnings(self, *args, **kwargs): pass
        def setup(self, *args, **kwargs): pass
        def input(self, *args, **kwargs): return self.HIGH
        def output(self, *args, **kwargs): pass
        def cleanup(self, *args, **kwargs): pass

    GPIO = MockGPIO()

# =====================================================
# PINOS DO MOTOR E BOTÕES
# =====================================================
MOTOR_LEFT_FWD = 17
MOTOR_LEFT_BWD = 18
MOTOR_RIGHT_FWD = 22
MOTOR_RIGHT_BWD = 23

# =====================================================
# CLASSES PRINCIPAIS
# =====================================================

class MotorDriver:
    """Controla os motores do robô."""
    def __init__(self):
        GPIO.setup(MOTOR_LEFT_FWD, GPIO.OUT)
        GPIO.setup(MOTOR_LEFT_BWD, GPIO.OUT)
        GPIO.setup(MOTOR_RIGHT_FWD, GPIO.OUT)
        GPIO.setup(MOTOR_RIGHT_BWD, GPIO.OUT)
        self.stop()

    def set_motor(self, left_speed, right_speed):
        """Define velocidades dos motores"""
        left_speed = int(np.clip(left_speed, -100, 100))
        right_speed = int(np.clip(right_speed, -100, 100))

        # Motor esquerdo
        GPIO.output(MOTOR_LEFT_FWD, GPIO.HIGH if left_speed > 0 else GPIO.LOW)
        GPIO.output(MOTOR_LEFT_BWD, GPIO.HIGH if left_speed < 0 else GPIO.LOW)

        # Motor direito
        GPIO.output(MOTOR_RIGHT_FWD, GPIO.HIGH if right_speed > 0 else GPIO.LOW)
        GPIO.output(MOTOR_RIGHT_BWD, GPIO.HIGH if right_speed < 0 else GPIO.LOW)

    def stop(self):
        """Para os motores"""
        GPIO.output(MOTOR_LEFT_FWD, GPIO.LOW)
        GPIO.output(MOTOR_LEFT_BWD, GPIO.LOW)
        GPIO.output(MOTOR_RIGHT_FWD, GPIO.LOW)
        GPIO.output(MOTOR_RIGHT_BWD, GPIO.LOW)


class LineFollowerController:
    """Controlador PD para seguir linha"""
    def __init__(self, config):
        self.max_speed = config["robot"]["max_speed"]
        self.base_speed = config["robot"]["base_speed"]
        self.k_gain = config["robot"]["k_gain"]
        self.k_derivative = config["robot"]["k_derivative"]
        self.dead_zone = config["robot"]["dead_zone"]
        self.error_history = deque(maxlen=5)
        self.last_error = 0

    def compute_speeds(self, centroid, width):
        if centroid is None:
            return 0, 0

        # Erro baseado na posição horizontal do centro da linha
        error = centroid[0] - (width // 2)
        self.error_history.append(error)
        avg_error = sum(self.error_history) / len(self.error_history)
        derivative = error - self.last_error
        self.last_error = error

        correction = int(self.k_gain * avg_error + self.k_derivative * derivative)

        left_speed = self.base_speed - correction
        right_speed = self.base_speed + correction

        # Evita pequenos ajustes desnecessários
        if abs(avg_error) < self.dead_zone:
            left_speed = right_speed = self.base_speed

        return (
            int(np.clip(left_speed, -self.max_speed, self.max_speed)),
            int(np.clip(right_speed, -self.max_speed, self.max_speed))
        )


class Camera:
    """Gerencia a captura de vídeo"""
    def __init__(self, width, height):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Câmera não encontrada ou em uso.")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Falha ao capturar frame da câmera.")
        return frame

    def release(self):
        self.cap.release()


class LineFollowerRobot:
    """Classe principal do robô seguidor de linha"""
    def __init__(self):
        self.camera = Camera(CAM_WIDTH, CAM_HEIGHT)
        self.motor = MotorDriver()
        self.controller = LineFollowerController(CONFIG)
        self.running = False
        self.thread = None

    def _process_frame(self, frame):
        """Processa a imagem e retorna o centro da linha"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = bw[CAM_HEIGHT - 80:CAM_HEIGHT - 20, :]
        M = cv2.moments(roi)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy + (CAM_HEIGHT - 80))

    def _loop(self):
        print("[INFO] Loop do robô iniciado.")
        while self.running:
            try:
                frame = self.camera.read()
                centroid = self._process_frame(frame)
                left_speed, right_speed = self.controller.compute_speeds(centroid, CAM_WIDTH)
                self.motor.set_motor(left_speed, right_speed)
            except Exception as e:
                print(f"[ERRO] {e}")
                self.motor.stop()
                time.sleep(0.1)

        self.motor.stop()
        print("[INFO] Loop do robô encerrado.")

    def start(self):
        if self.running:
            print("[AVISO] Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[INFO] Robô iniciado.")

    def stop(self):
        if not self.running:
            print("[AVISO] Robô já está parado.")
            return
        self.running = False
        if self.thread:
            self.thread.join()
        self.motor.stop()
        print("[INFO] Robô parado.")

    def cleanup(self):
        self.motor.stop()
        self.camera.release()
        GPIO.cleanup()

# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():
    print("[INFO] Iniciando main.py")
    robot = LineFollowerRobot()

    try:
        print("[INFO] Pressione 's' + Enter para START, 'x' + Enter para STOP, Ctrl+C para sair.")
        while True:
            cmd = input("Comando: ").strip().lower()
            if cmd == 's':
                robot.start()
            elif cmd == 'x':
                robot.stop()
            else:
                print("[ERRO] Comando inválido. Use 's' para start, 'x' para stop.")
    except KeyboardInterrupt:
        print("\n[INFO] Encerrando programa...")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()

