# main.py corrigido
# Segue-linha robusto + UI web + botão físico.
# Ajustes:
#  - Logs claros para detecção de verde
#  - Tempos de curva fáceis de calibrar
#  - Giro menos brusco (bias reduzido)

import os
import cv2
import time
import json
import threading
import numpy as np
import math
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

    BASE_SPEED = 20
    MIX_ANGLE = 0.7
    MAX_ANGLE = 50.0

    N_STRIPS = 5
    STRIP_H = 28
    STRIP_BOTTOM = 440

    INTERSECT_DEBOUNCE = 6
    C90_DEBOUNCE = 6
    GREEN_DEBOUNCE = 2

    # tempos calibráveis
    INTERSECT_FWD_TIME = 0.8
    TURN90_FWD_TIME = 0.4
    TURN90_TURN_TIME = 0.85   # ajustável (era 0.9)

    PID_DEFAULTS = {"kp": 0.6, "ki": 0.0, "kd": 0.1, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=False)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        self.history = deque(maxlen=5)
        self._intersect_seen = 0
        self._c90_seen = 0
        self._green_seen = 0
        self._green_last = None

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
        try: self.hardware.stop()
        except Exception: pass
        log("Parado.")

    def _drive(self, base_speed: float, error: float):
        try:
            self.hardware.set_motor_speed(base_speed, error)
        except Exception:
            pass

    def _forward_time(self, duration_s: float):
        self.hardware.set_motor_speed(self.BASE_SPEED, 0)
        start_time = time.time()
        while self.running and time.time() - start_time < duration_s:
            time.sleep(0.01)
        self.hardware.stop()

    def _turn_in_place_time(self, direction: str, duration_s: float):
        bias = 100 if direction == "left" else -100  # menos agressivo
        start_time = time.time()
        while self.running and time.time() - start_time < duration_s:
            self.hardware.set_motor_speed(0, bias)
            time.sleep(0.01)
        self.hardware.stop()

    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()
                bw = self._binarize(frame)

                cents, widths = [], []
                for i in range(self.N_STRIPS):
                    y0 = self.STRIP_BOTTOM - i*self.STRIP_H
                    c, w = self._strip_centroid(bw, y0, self.STRIP_H)
                    cents.append(c)
                    widths.append(w)

                valids = [p for p in cents if p is not None]
                angle = self._fit_angle(valids)

                is_intersection, is_curve90, _ = self._detect_intersection(bw, widths, cents, angle)

                if is_curve90:
                    self._c90_seen += 1
                else:
                    self._c90_seen = 0
                confirmed_curve90 = self._c90_seen >= self.C90_DEBOUNCE

                if is_intersection:
                    self._intersect_seen += 1
                else:
                    self._intersect_seen = 0
                confirmed_intersection = self._intersect_seen >= self.INTERSECT_DEBOUNCE

                green_centroids, green_dir = self.vision.detect_greens(frame)
                if green_dir:
                    self._green_last = green_dir
                    self._green_seen += 1
                else:
                    self._green_seen = 0
                confirmed_green = self._green_last if self._green_seen >= self.GREEN_DEBOUNCE else None

                # logs
                if confirmed_green:
                    log(f"Marcador verde confirmado: {confirmed_green}")

                if confirmed_intersection and not confirmed_curve90:
                    if confirmed_green == "uturn":
                        self._forward_time(self.INTERSECT_FWD_TIME)
                        self._turn_in_place_time("left", self.TURN90_TURN_TIME * 2)
                        log("Execução: U-Turn à esquerda")
                        self._green_seen = 0; self._intersect_seen = 0; self._green_last = None
                        continue
                    elif confirmed_green in ("left", "right"):
                        self._forward_time(self.TURN90_FWD_TIME)
                        self._turn_in_place_time(confirmed_green, self.TURN90_TURN_TIME)
                        log(f"Execução: curva {confirmed_green}")
                        self._green_seen = 0; self._intersect_seen = 0; self._green_last = None
                        continue
                    else:
                        self._forward_time(self.INTERSECT_FWD_TIME)
                        log("Execução: interseção sem verde → reto")
                        self._intersect_seen = 0
                        continue

                if valids:
                    offset = valids[0][0] - (self.WIDTH // 2)
                    error = float(offset) + self.MIX_ANGLE * float(angle)
                    self._drive(self.BASE_SPEED, error)
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
            try: self.hardware.stop()
            except Exception: pass

    # ==== visão utilitários ====
    def _binarize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 7)
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

    def _fit_angle(self, pts):
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        alpha, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        angle_deg = np.degrees(np.arctan2(alpha, 1.0))
        return float(np.clip(angle_deg, -self.MAX_ANGLE, self.MAX_ANGLE))

    def _detect_intersection(self, bw, widths, cents, angle_deg):
        bottom_widths = [w for w in widths[:3] if w > 0]
        is_wide = len(bottom_widths) > 0 and np.mean(bottom_widths) > 250
        is_intersection = is_wide and abs(angle_deg) < 15

        num_valid_cents = sum(1 for c in cents if c is not None)
        is_curve90 = False
        if num_valid_cents >= 3:
            bottom_cx = cents[0][0] if cents[0] is not None else self.WIDTH // 2
            top_cx = cents[-1][0] if cents[-1] is not None else self.WIDTH // 2
            shift_px = abs(bottom_cx - top_cx)
            is_curve90 = (shift_px > 80) and (abs(angle_deg) > 20)
        elif num_valid_cents >= 2:
            is_curve90 = abs(angle_deg) > 25

        return is_intersection, is_curve90, False

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

