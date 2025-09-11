#!/usr/bin/env python3
"""
Robô seguidor de linha — versão com adaptadores para Vision/Hardware faltantes.
Mantém funcionalidade: captura, binarização, centróide, PD, envio ao hardware, botão físico.
"""

import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
import sys, traceback

# ---------- Config ----------
USE_LEDS = True
AUTO_START = True
CANDIDATE_BUTTON_PINS = [21, 4]

CONFIG_DEFAULTS = {
    "camera": {"width": 640, "height": 480, "rotate_180": False},
    "robot": {
        "max_speed": 100,
        "base_speed": 60,
        "k_gain": 0.02,
        "k_derivative": 0.01,
        "dead_zone": 10
    }
}

# ---------- Logging / shared state ----------
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "config": {},
            "last_frame": None,
            "speeds": {"left": 0, "right": 0},
            "view_mode": "preview",
            "status": "idle",
            "fps": 0.0,
            "log": []
        }

    def get(self, key):
        with self._lock:
            return self._state.get(key)

    def set(self, key, value):
        with self._lock:
            self._state[key] = value

    def append_log(self, msg: str):
        with self._lock:
            stamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {msg}"
            self._state["log"].append(line)
            if len(self._state["log"]) > 300:
                self._state["log"] = self._state["log"][-300:]
            print(line, flush=True)

SHARED_STATE = SharedState()

def log(msg: str):
    SHARED_STATE.append_log(msg)

def _excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    log(f"Unhandled exception: {exc_type.__name__}: {exc}")
sys.excepthook = _excepthook

# ---------- Optional hardware imports (safe) ----------
GPIO_AVAILABLE = True
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except Exception as e:
    log(f"RPi.GPIO indisponível: {e}")
    GPIO_AVAILABLE = False

PICAMERA_AVAILABLE = True
Transform = None
try:
    from picamera2 import Picamera2
    try:
        from libcamera import Transform as _T
        Transform = _T
    except Exception:
        Transform = None
except Exception as e:
    log(f"picamera2 não disponível: {e}")
    PICAMERA_AVAILABLE = False

try:
    from hardware_control import HardwareControl
except Exception as e:
    log(f"hardware_control não disponível: {e}")
    HardwareControl = None

try:
    from vision import Vision
except Exception as e:
    log(f"vision não disponível: {e}")
    Vision = None

try:
    from led_control import LedController
except Exception as e:
    log(f"led_control não disponível: {e}")
    LedController = None

# ---------- Camera abstraction ----------
class Camera:
    def __init__(self, width: int, height: int, rotate_180: bool):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                main_cfg = {"size": (self.width, self.height), "format": "RGB888"}
                if Transform is not None:
                    cfg = self.picam.create_video_configuration(main=main_cfg, transform=Transform(hflip=0, vflip=0), buffer_count=3)
                else:
                    cfg = self.picam.create_video_configuration(main=main_cfg, buffer_count=3)
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
            self._init_usb_camera()

    def _init_usb_camera(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Nenhuma câmera USB disponível.")
            log("Câmera USB inicializada.")
        except Exception as e:
            log(f"Erro ao inicializar câmera USB: {e}")
            raise

    def _reconnect_usb(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self._init_usb_camera()
            log("Câmera USB reconectada.")
        except Exception as e:
            log(f"Falha ao reconectar USB: {e}")
            raise RuntimeError("Falha ao reconectar câmera USB.")

    def read(self):
        for attempt in range(3):
            try:
                if self.picam is not None:
                    rgb = self.picam.capture_array()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = self.cap.read()
                    if not ok:
                        self._reconnect_usb()
                        continue
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                return frame
            except Exception as e:
                log(f"Erro lendo câmera (tentativa {attempt+1}/3): {e}")
                time.sleep(0.1)
        raise RuntimeError("Falha persistente na câmera.")

    def release(self):
        try:
            if self.picam is not None:
                try:
                    self.picam.stop()
                except Exception:
                    pass
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            log("Recursos da câmera liberados.")
        except Exception as e:
            log(f"Erro ao liberar câmera: {e}")

# ---------- Controller PD ----------
class LineFollowerController:
    def __init__(self, cfg: Dict):
        self.max_speed = cfg["robot"]["max_speed"]
        self.base_speed = cfg["robot"]["base_speed"]
        self.k_gain = cfg["robot"]["k_gain"]
        self.k_derivative = cfg["robot"]["k_derivative"]
        self.dead_zone = cfg["robot"]["dead_zone"]
        self.error_history = deque(maxlen=5)
        self.last_error = None

    def compute_speeds(self, centroid: Optional[Tuple[int,int]], width: int) -> Tuple[int,int]:
        if centroid is None:
            self.error_history.clear()
            self.last_error = None
            return 0, 0
        error = centroid[0] - (width // 2)
        self.error_history.append(error)
        avg_error = sum(self.error_history) / len(self.error_history) if self.error_history else error
        derivative = (error - self.last_error) if (self.last_error is not None) else 0
        self.last_error = error
        correction = int(self.k_gain * avg_error + self.k_derivative * derivative)
        left = self.base_speed - correction
        right = self.base_speed + correction
        if abs(avg_error) < self.dead_zone:
            left = right = self.base_speed
        left = int(np.clip(left, -self.max_speed, self.max_speed))
        right = int(np.clip(right, -self.max_speed, self.max_speed))
        return left, right

# ---------- Robot with adapters ----------
class Robot:
    def __init__(self, config: Dict = CONFIG_DEFAULTS, allow_stubs: bool = True):
        self.config = config

        # Camera
        try:
            self.camera = Camera(**self.config["camera"])
        except Exception as e:
            log(f"Erro ao iniciar Camera: {e}")
            if not allow_stubs:
                raise
            class _StubCam:
                def __init__(self, w,h):
                    self.width = w; self.height = h
                def read(self):
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                def release(self): pass
            self.camera = _StubCam(self.config["camera"]["width"], self.config["camera"]["height"])
            log("Usando stub de câmera para depuração.")

        # Vision: instancia se possível
        if Vision is not None:
            try:
                self.vision = Vision({}, log)
            except Exception as e:
                log(f"Erro ao instanciar Vision: {e}")
                self.vision = None
        else:
            self.vision = None

        # ADAPTADOR: garantir strip_centroid
        if not hasattr(self.vision, "strip_centroid"):
            # tenta detectar métodos alternativos no objeto Vision (se existir)
            adapted = False
            if self.vision is not None:
                candidates = [
                    "find_centroid", "find_line_centroid", "centroid", "find_centers",
                    "get_centroid", "compute_centroid"
                ]
                for name in candidates:
                    if hasattr(self.vision, name):
                        func = getattr(self.vision, name)
                        def _wrap(bw, y, h, _f=func):
                            try:
                                return _f(bw, y, h)
                            except TypeError:
                                # se assinatura diferente (por exemplo, apenas bw), tente adaptar
                                try:
                                    return _f(bw)
                                except Exception:
                                    return None, bw[y:y+h,:]
                        self.vision.strip_centroid = _wrap
                        log(f"Adaptado Vision: usando método '{name}' como strip_centroid.")
                        adapted = True
                        break
            if not adapted:
                # fallback: implementa strip_centroid por momentos
                log("Vision sem strip_centroid — usando fallback baseado em momentos (cv2.moments).")
                def _fallback_strip_centroid(bw, y, h):
                    roi = bw[y:y+h, :]
                    try:
                        M = cv2.moments(roi)
                    except Exception:
                        return None, roi
                    if M and M.get("m00", 0) > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"]) + y
                        return (cx, cy), roi
                    return None, roi
                class _VStub:
                    def strip_centroid(self, bw, y, h):
                        return _fallback_strip_centroid(bw, y, h)
                self.vision = _VStub()

        # HardwareControl: instancia se possível
        if HardwareControl is not None:
            try:
                self.hardware = HardwareControl({"pid": {}})
            except Exception as e:
                log(f"Erro ao instanciar HardwareControl: {e}")
                self.hardware = None
        else:
            self.hardware = None

        # ADAPTADOR: garantir drive e stop
        if not hasattr(self.hardware, "drive"):
            adapted_hw = False
            if self.hardware is not None:
                # procura por métodos parecidos e adapta
                if hasattr(self.hardware, "set_motors"):
                    def _drive_map(l, r):
                        try:
                            self.hardware.set_motors(left=l, right=r)
                        except Exception:
                            try:
                                self.hardware.set_motors(l, r)
                            except Exception:
                                log("Falha ao chamar hardware.set_motors via adapter.")
                    self.hardware.drive = _drive_map
                    adapted_hw = True
                    log("Adaptado HardwareControl: usando set_motors -> drive.")
                elif hasattr(self.hardware, "set_speed"):
                    def _drive_map(l, r):
                        try:
                            self.hardware.set_speed({"left": l, "right": r})
                        except Exception:
                            log("Falha ao chamar hardware.set_speed via adapter.")
                    self.hardware.drive = _drive_map
                    adapted_hw = True
                    log("Adaptado HardwareControl: usando set_speed -> drive.")
                elif hasattr(self.hardware, "write_pwm"):
                    def _drive_map(l, r):
                        try:
                            # tenta chamar com possíveis assinaturas
                            try:
                                self.hardware.write_pwm(left_cmd=float(l), right_cmd=float(r))
                            except TypeError:
                                try:
                                    self.hardware.write_pwm(l, r)
                                except Exception:
                                    log("Falha ao chamar hardware.write_pwm via adapter.")
                        except Exception:
                            log("Erro no adapter write_pwm -> drive.")
                    self.hardware.drive = _drive_map
                    adapted_hw = True
                    log("Adaptado HardwareControl: usando write_pwm -> drive.")
            if not adapted_hw:
                # fallback stub que apenas loga — protege o loop
                log("HardwareControl sem drive — usando stub que loga comandos (evita crash).")
                class _HWStub:
                    def drive(self, l, r):
                        log(f"[HW-STUB] drive chamado: left={l} right={r}")
                    def stop(self):
                        log("[HW-STUB] stop chamado")
                self.hardware = _HWStub()
        else:
            # hardware tem drive: garantir stop também
            if not hasattr(self.hardware, "stop"):
                def _stop_stub():
                    log("[HW-ADAPTER] stop não implementado pelo hardware; adapter noop.")
                self.hardware.stop = _stop_stub

        # LEDs (opcional)
        self.leds = None
        if USE_LEDS and LedController is not None:
            try:
                self.leds = LedController()
            except Exception as e:
                log(f"Falha ao iniciar LedController: {e}")
                self.leds = None

        # Controller e estado
        self.controller = LineFollowerController(self.config)
        self.width = self.config["camera"]["width"]
        self.height = self.config["camera"]["height"]

        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            log("Robô já rodando.")
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        SHARED_STATE.set("status", "running")
        log("Loop principal iniciado.")

    def stop(self):
        if not self.running:
            log("Stop chamado, mas robô já parado.")
        self.running = False
        SHARED_STATE.set("status", "stopped")
        try:
            self.hardware.stop()
            log("Motores parados.")
        except Exception as e:
            log(f"Erro ao parar motores: {e}")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            log("Thread principal encerrada.")

    def _loop(self):
        target_fps = 30.0
        target_period = 1.0 / target_fps
        try:
            while self.running:
                t0 = time.time()
                frame = self.camera.read()
                try:
                    bw = self._binarize(frame)
                except Exception as e:
                    log(f"Falha na binarização: {e}")
                    bw = np.zeros((self.height, self.width), dtype=np.uint8)

                centroid = None
                try:
                    centroid, roi = self.vision.strip_centroid(bw, self.height - 50, 40)
                except Exception as e:
                    log(f"Erro em vision.strip_centroid: {e}")
                    centroid = None

                left_speed, right_speed = self.controller.compute_speeds(centroid, self.width)

                try:
                    self.hardware.drive(left_speed, right_speed)
                except Exception as e:
                    log(f"Erro enviando comandos ao hardware: {e}")

                SHARED_STATE.set("last_frame", frame)
                SHARED_STATE.set("speeds", {"left": left_speed, "right": right_speed})

                elapsed = time.time() - t0
                sleep_time = max(0, target_period - elapsed)
                time.sleep(sleep_time)
                total = elapsed + sleep_time if (elapsed + sleep_time) > 0 else 1e-6
                SHARED_STATE.set("fps", 1.0 / total)
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass

    def _binarize(self, frame_bgr):
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            return bw
        except Exception as e:
            log(f"Erro na binarização: {e}")
            raise

    def release(self):
        try:
            self.camera.release()
        except Exception:
            pass
        log("Recursos do robô liberados.")

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop()
        finally:
            self.release()

# ---------- GPIO button setup ----------
def _setup_button(robot: Robot):
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: não configurando botão físico.")
        return
    try:
        for pin in CANDIDATE_BUTTON_PINS:
            try:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            except Exception as e:
                log(f"Falha ao configurar GPIO {pin}: {e}")
                continue

            last_state = GPIO.input(pin)
            last_change = time.time()

            def _toggle(channel, _pin=pin):
                nonlocal last_state, last_change
                try:
                    current_state = GPIO.input(channel)
                    now = time.time()
                    if current_state == GPIO.LOW and last_state == GPIO.HIGH and (now - last_change) > 0.3:
                        if robot.running:
                            robot.stop()
                        else:
                            robot.start()
                        last_change = now
                    last_state = current_state
                except Exception as e:
                    log(f"Erro no callback do botão BCM {channel}: {e}")

            GPIO.add_event_detect(pin, GPIO.BOTH, callback=_toggle, bouncetime=300)
            log(f"Botão físico pronto no BCM {pin}.")
            break
    except Exception as e:
        log(f"Erro ao configurar botões físicos: {e}")

# ---------- Main ----------
def main():
    try:
        with Robot(CONFIG_DEFAULTS, allow_stubs=True) as robot:
            _setup_button(robot)
            if AUTO_START:
                robot.start()
            try:
                log("Loop principal (pressione Ctrl+C para sair).")
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                log("Interrupção (Ctrl+C) recebida.")
                robot.stop()
    except Exception as e:
        log(f"Erro fatal na inicialização: {e}")
        raise

if __name__ == "__main__":
    main()
