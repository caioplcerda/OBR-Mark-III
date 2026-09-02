# main_fixed.py
# Segue-linha robusto + UI web + botão físico (corrigido)
# Ajustado para seguir mais reto com controle proporcional (P-controller simples)

import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque

USE_LEDS = True

WEB_AVAILABLE = True
SHARED_STATE = {
    "config": {},
    "last_frame": None,
    "speeds": {"left": 0, "right": 0},
    "view_mode": "preview",
    "status": "idle",
    "fps": 0.0,
    "log": [],
}
try:
    import web_stream
    if hasattr(web_stream, "SHARED_STATE"):
        SHARED_STATE = web_stream.SHARED_STATE
    WEB_AVAILABLE = True
except Exception:
    WEB_AVAILABLE = False

GPIO_AVAILABLE = True
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    try:
        GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.cleanup(21)
    except Exception:
        GPIO_AVAILABLE = False
except Exception:
    GPIO_AVAILABLE = False

CANDIDATE_BUTTON_PINS = [21, 4]

from hardware_control import HardwareControl
from vision import Vision
try:
    from led_control import LedController
except Exception:
    LedController = None

PICAMERA_AVAILABLE = True
try:
    from picamera2 import Picamera2
    try:
        from libcamera import Transform
    except Exception:
        Transform = None
    PICAMERA_AVAILABLE = True
except Exception:
    PICAMERA_AVAILABLE = False


def log(msg: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        SHARED_STATE["log"].append(line)
        if len(SHARED_STATE["log"]) > 300:
            SHARED_STATE["log"] = SHARED_STATE["log"][-300:]
    except Exception:
        pass


class Camera:
    def __init__(self, width=640, height=480, rotate_180=False):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None
        try:
            import cv2 as _cv2
            _cv2.setNumThreads(1)
        except Exception:
            pass

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                if Transform is not None:
                    cfg = self.picam.create_video_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"},
                        transform=Transform(hflip=0, vflip=0),
                        buffer_count=3
                    )
                else:
                    cfg = self.picam.create_video_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"},
                        buffer_count=3
                    )
                self.picam.configure(cfg)
                try:
                    self.picam.set_controls({"FrameDurationLimits": (10000, 33333)})
                except Exception:
                    pass
                self.picam.start()
                log("Picamera2 iniciada em modo vídeo.")
            except Exception as e:
                log(f"Falha Picamera2: {e}. Usando OpenCV/USB.")
                self.picam = None

        if self.picam is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Nenhuma câmera disponível.")

    def read(self):
        if self.picam is not None:
            rgb = self.picam.capture_array()
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ok, frame_bgr = self.cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler frame da câmera USB.")
        if self.rotate_180:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        return frame_bgr

    def release(self):
        try:
            if self.picam is not None:
                self.picam.stop()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class Robot:
    WIDTH = 640
    HEIGHT = 480

    MAX_SPEED = 100
    BASE_SPEED = 60     # menor que o máximo → dá margem de correção
    K_GAIN = 0.02       # ajuste do controlador proporcional
    DEAD_ZONE = 10      # zona morta para evitar ziguezague

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=False)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": {}})

        self.history = deque(maxlen=5)

    def start(self):
        if self.running:
            log("Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        SHARED_STATE["status"] = "running"
        log("Loop principal iniciado.")

    def stop(self):
        self.running = False
        SHARED_STATE["status"] = "stopped"
        try:
            self.hardware.stop()
        except Exception:
            pass
        log("Parado.")

    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()
                bw = self._binarize(frame)

                centroid, _ = self._strip_centroid(bw, self.HEIGHT - 50, 40)

                if centroid is not None:
                    error = centroid[0] - (self.WIDTH // 2)

                    # controle proporcional
                    correction = int(self.K_GAIN * error)

                    left = self.BASE_SPEED - correction
                    right = self.BASE_SPEED + correction

                    # aplica zona morta
                    if abs(error) < self.DEAD_ZONE:
                        left = right = self.BASE_SPEED

                    # limita velocidades
                    left = int(np.clip(left, -self.MAX_SPEED, self.MAX_SPEED))
                    right = int(np.clip(right, -self.MAX_SPEED, self.MAX_SPEED))

                    self.hardware.drive(left, right)
                else:
                    self.hardware.stop()

                SHARED_STATE["last_frame"] = frame
                SHARED_STATE["speeds"] = {
                    "left": self.hardware.last_left_speed,
                    "right": self.hardware.last_right_speed
                }
                time.sleep(0.01)
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass

    # ==== visão utilitários ====
    def _binarize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 7
        )
        return bw

    def _strip_centroid(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, :]
        colsum = roi.sum(axis=0)
        if colsum.max() < 255 * h * 0.05:
            return None, 0
        x = np.arange(0, W, dtype=np.float32)
        cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
        cy = int((y0 + y1) / 2)
        return (int(cx), int(cy)), int(np.count_nonzero(colsum))


def _wire_web(robot: Robot):
    if not WEB_AVAILABLE:
        return
    try:
        if hasattr(web_stream, "register_robot"):
            web_stream.register_robot(robot)
            log("Robô registrado no servidor web.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")


def _setup_button(robot: Robot):
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: sem botão físico.")
        return
    for pin in [21, 4]:
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            def _toggle(channel):
                if GPIO.input(pin) == GPIO.LOW:
                    if robot.running:
                        robot.stop()
                    else:
                        robot.start()
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=_toggle, bouncetime=300)
            log(f"Botão físico pronto no BCM {pin}.")
            break
        except Exception as e:
            log(f"Falha botão BCM {pin}: {e}")


def main():
    robot = Robot()
    _wire_web(robot)
    _setup_button(robot)

    if WEB_AVAILABLE and hasattr(web_stream, "app"):
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5000"))
        try:
            if hasattr(web_stream, "socketio"):
                web_stream.socketio.run(web_stream.app, host=host, port=port, allow_unsafe_werkzeug=True)
            else:
                web_stream.app.run(host=host, port=port)
        finally:
            robot.stop()
    else:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            robot.stop()


if __name__ == "__main__":
    main()
