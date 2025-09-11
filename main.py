import os
import cv2
import time
import json
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

    BASE_SPEED = 30
    TURN_SPEED = 18
    MIX_ANGLE = 0.7
    MAX_ANGLE = 50.0

    N_STRIPS = 5
    STRIP_H = 28
    STRIP_BOTTOM = 440

    INTERSECT_DEBOUNCE = 6
    INTERSECT_AHEAD_DEBOUNCE = 4
    C90_DEBOUNCE = 6
    GREEN_DEBOUNCE = 2

    INTERSECT_FWD_TIME = 0.7
    TURN90_FWD_TIME = 0.3
    TURN90_TURN_TIME = 0.75

    TURN_BIAS = 60
    TURN_BIAS_SIGN = 1
    TURN_DIRECTION_INVERT = False

    PID_DEFAULTS = {"kp": 0.6, "ki": 0.0, "kd": 0.1, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=False)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        self.history = deque(maxlen=5)
        self._intersect_seen = 0
        self._intersect_ahead_seen = 0
        self._c90_seen = 0
        self._green_seen = 0
        self._green_last = None
        self.planned_direction = None

    # ciclo de vida
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
        log("Parado (stop chamado).")

    # controle
    def _drive(self, base_speed: float, error: float):
        try:
            self.hardware.set_motor_speed(base_speed, error)
        except Exception:
            pass

    def _forward_time(self, duration_s: float, reason: str = "forward"):
        log(f"➡️ Avançando {duration_s:.2f}s | motivo={reason}")
        self.hardware.set_motor_speed(self.BASE_SPEED, 0)
        start_time = time.time()
        while self.running and (time.time() - start_time < duration_s):
            time.sleep(0.01)
        self.hardware.stop()
        log(f"✅ Avanço concluído | motivo={reason}")

    # utilitários de direção
    def set_turn_direction_invert(self, invert: bool):
        self.TURN_DIRECTION_INVERT = bool(invert)
        log(f"Configuração: TURN_DIRECTION_INVERT = {self.TURN_DIRECTION_INVERT}")

    def toggle_turn_direction_invert(self):
        self.TURN_DIRECTION_INVERT = not self.TURN_DIRECTION_INVERT
        log(f"Toggled: TURN_DIRECTION_INVERT = {self.TURN_DIRECTION_INVERT}")

    def set_turn_bias_sign(self, sign: int):
        self.TURN_BIAS_SIGN = 1 if sign >= 0 else -1
        log(f"Configuração: TURN_BIAS_SIGN = {self.TURN_BIAS_SIGN}")

    def _compute_turn_params(self, direction: str, arc: bool = True):
        requested = (direction or "left").lower()
        if requested not in ("left", "right"):
            requested = "left"

        applied = requested if not self.TURN_DIRECTION_INVERT else ("left" if requested == "right" else "right")

        if arc:
            base = int(self.TURN_SPEED * 0.6)
        else:
            base = 0

        raw_bias = self.TURN_BIAS if applied == "left" else -self.TURN_BIAS
        bias = raw_bias * self.TURN_BIAS_SIGN

        if arc:
            bias = int(bias * 0.75)

        log(f"[TURN PARAMS] req={requested} -> applied={applied} | base={base} | bias={bias} (arc={arc})")
        return requested, applied, base, bias

    def _turn_in_place_time(self, direction: str, duration_s: float, reason: str = "turn", arc: bool = True):
        requested, applied, base, bias = self._compute_turn_params(direction, arc=arc)

        log(f"↪️ Iniciando TURN | requested={requested} | applied={applied} | duration={duration_s:.2f}s | reason={reason}")
        start = time.time()
        try:
            while self.running and (time.time() - start < duration_s):
                self.hardware.set_motor_speed(base, bias)
                time.sleep(0.01)
        finally:
            self.hardware.stop()
            log(f"✅ TURN finalizado | applied={applied} | reason={reason}")

    def test_turn_sign(self, duration: float = 0.35, small_bias: int = 30):
        log("🔧 Teste de direção de giro iniciado. Observe o lado para +bias e -bias.")
        try:
            log(f"Teste PART 1: bias = +{small_bias}")
            self.hardware.set_motor_speed(0, int(small_bias))
            time.sleep(duration)
            self.hardware.stop()
            time.sleep(0.15)

            log(f"Teste PART 2: bias = -{small_bias}")
            self.hardware.set_motor_speed(0, int(-small_bias))
            time.sleep(duration)
            self.hardware.stop()
            log("🔧 Teste concluído. Ajuste TURN_BIAS_SIGN ou TURN_DIRECTION_INVERT se necessário.")
        except Exception as e:
            log(f"Erro no teste de giro: {e}")
            try:
                self.hardware.stop()
            except Exception:
                pass

    # eventos de visão
    def on_intersection(self):
        log("🟦 Interseção detectada!")
        self._forward_time(self.INTERSECT_FWD_TIME, reason="intersection")

    def on_curve_90(self, direction="left"):
        log(f"🟨 Curva de 90° detectada ({direction})")
        self._forward_time(self.TURN90_FWD_TIME, reason="curve_prep")
        self._turn_in_place_time(direction, self.TURN90_TURN_TIME, reason="curve_90", arc=True)

    def on_green_marker(self):
        log("🟩 Marcador verde detectado!")
        self._forward_time(0.5, reason="green_marker")

    # loop principal (esqueleto, depende da Vision)
    def _loop(self):
        while self.running:
            frame = self.camera.read()
            result = self.vision.process(frame)
            # aqui você chama on_intersection(), on_curve_90(), on_green_marker()
            # dependendo de result
            time.sleep(0.01)
