import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple

USE_LEDS = True
WEB_AVAILABLE = False  # Desativado temporariamente para depuração
GPIO_AVAILABLE = True

try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except Exception as e:
    print(f"Erro ao importar RPi.GPIO: {e}")
    GPIO_AVAILABLE = False

CANDIDATE_BUTTON_PINS = [21, 4]

try:
    from hardware_control import HardwareControl
    from vision import Vision
except ImportError as e:
    print(f"Erro ao importar módulos de hardware ou visão: {e}")
    HardwareControl = None
    Vision = None

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
except Exception as e:
    print(f"Erro ao importar picamera2: {e}")
    PICAMERA_AVAILABLE = False


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

    def get(self, key: str):
        with self._lock:
            return self._state.get(key)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._state[key] = value

    def append_log(self, msg: str) -> None:
        with self._lock:
            stamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {msg}"
            self._state["log"].append(line)
            if len(self._state["log"]) > 300:
                self._state["log"] = self._state["log"][-300:]
            print(line, flush=True)


SHARED_STATE = SharedState()


def log(msg: str) -> None:
    SHARED_STATE.append_log(msg)


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


class Camera:
    def __init__(self, width: int, height: int, rotate_180: bool):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None
        try:
            cv2.setNumThreads(1)
        except Exception as e:
            log(f"Erro ao configurar OpenCV threads: {e}")

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                cfg_kwargs = {
                    "main": {"size": (self.width, self.height), "format": "RGB888"},
                    "buffer_count": 3
                }
                if Transform is not None:
                    cfg = self.picam.create_video_configuration(
                        main=cfg_kwargs["main"],
                        transform=Transform(hflip=0, vflip=0),
                        buffer_count=cfg_kwargs["buffer_count"]
                    )
                else:
                    cfg = self.picam.create_video_configuration(
                        main=cfg_kwargs["main"],
                        buffer_count=cfg_kwargs["buffer_count"]
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
            self._init_usb_camera()

    def _init_usb_camera(self) -> None:
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Nenhuma câmera USB disponível.")
            log("Câmera USB inicializada.")
        except Exception as e:
            log(f"Erro ao inicializar câmera USB: {e}")
            raise

    def _reconnect_usb(self) -> None:
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self._init_usb_camera()
            log("Câmera USB reconectada com sucesso.")
        except Exception as e:
            log(f"Falha ao reconectar câmera USB: {e}")
            raise RuntimeError("Falha ao reconectar câmera USB.")

    def read(self) -> np.ndarray:
        for attempt in range(3):
            try:
                if self.picam is not None:
                    rgb = self.picam.capture_array()
                    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    ok, frame_bgr = self.cap.read()
                    if not ok:
                        self._reconnect_usb()
                        continue
                if self.rotate_180:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
                return frame_bgr
            except Exception as e:
                log(f"Erro ao ler frame (tentativa {attempt + 1}/3): {e}")
                time.sleep(0.1)
        raise RuntimeError("Falha persistente na câmera.")

    def release(self) -> None:
        try:
            if self.picam is not None:
                try:
                    self.picam.stop()
                    log("Picamera2 liberada.")
                except Exception as e:
                    log(f"Erro ao parar Picamera2: {e}")
            if self.cap is not None:
                try:
                    self.cap.release()
                    log("Câmera USB liberada.")
                except Exception as e:
                    log(f"Erro ao liberar câmera USB: {e}")
        except Exception as e:
            log(f"Erro ao liberar câmera: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class LineFollowerController:
    def __init__(self, config: Dict):
        self.max_speed: int = config["robot"]["max_speed"]
        self.base_speed: int = config["robot"]["base_speed"]
        self.k_gain: float = config["robot"]["k_gain"]
        self.k_derivative: float = config["robot"]["k_derivative"]
        self.dead_zone: int = config["robot"]["dead_zone"]
        self.error_history: deque = deque(maxlen=5)
        self.last_error: Optional[float] = None

    def compute_speeds(self, centroid: Optional[Tuple[int, int]], width: int) -> Tuple[int, int]:
        if centroid is None:
            self.error_history.clear()
            self.last_error = None
            return 0, 0
        error = centroid[0] - (width // 2)
        self.error_history.append(error)
        avg_error = sum(self.error_history) / len(self.error_history) if self.error_history else error
        derivative = (error - self.last_error) if self.last_error is not None else 0
        self.last_error = error
        correction = int(self.k_gain * avg_error + self.k_derivative * derivative)
        left = self.base_speed - correction
        right = self.base_speed + correction
        if abs(avg_error) < self.dead_zone:
            left = right = self.base_speed
        return (
            int(np.clip(left, -self.max_speed, self.max_speed)),
            int(np.clip(right, -self.max_speed, self.max_speed))
        )


class Robot:
    def __init__(self):
        if HardwareControl is None or Vision is None:
            raise RuntimeError("Módulos hardware_control ou vision não disponíveis.")
        config = CONFIG_DEFAULTS
        self.camera: Camera = Camera(**config["camera"])
        self.vision: Vision = Vision({}, log)

        # Fallback se Vision não tiver strip_centroid
        if not hasattr(self.vision, "strip_centroid"):
            log("Vision sem strip_centroid — usando fallback interno.")
            def _fallback_strip_centroid(bw, y, h):
                roi = bw[y:y+h, :]
                M = cv2.moments(roi)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + y
                    return (cx, cy), roi
                return None, roi
            self.vision.strip_centroid = _fallback_strip_centroid

        self.hardware: HardwareControl = HardwareControl({"pid": {}})
        self.controller: LineFollowerController = LineFollowerController(config)
        self.width: int = config["camera"]["width"]
        self.height: int = config["camera"]["height"]
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.running:
            log("Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        SHARED_STATE.set("status", "running")
        log("Loop principal iniciado.")

    def stop(self) -> None:
        self.running = False
        SHARED_STATE.set("status", "stopped")
        try:
            self.hardware.stop()
            log("Motores parados.")
        except Exception as e:
            log(f"Erro ao parar motores: {e}")
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
            log("Thread principal encerrada.")

    def _loop(self) -> None:
        target_fps = 30
        target_period = 1.0 / target_fps
        try:
            while self.running:
                start_time = time.time()
                frame = self.camera.read()
                bw = self._binarize(frame)
                centroid, _ = self.vision.strip_centroid(bw, self.height - 50, 40)
                left_speed, right_speed = self.controller.compute_speeds(centroid, self.width)
                try:
                    self.hardware.drive(left_speed, right_speed)
                except Exception as e:
                    log(f"Erro ao enviar comandos aos motores: {e}")
                SHARED_STATE.set("last_frame", frame)
                SHARED_STATE.set("speeds", {"left": left_speed, "right": right_speed})
                elapsed = time.time() - start_time
                sleep_time = max(0, target_period - elapsed)
                time.sleep(sleep_time)
                SHARED_STATE.set("fps", 1.0 / (elapsed + sleep_time) if (elapsed + sleep_time) > 0 else 0)
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass

    def _binarize(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            return bw
        except Exception as e:
            log(f"Erro ao binarizar imagem: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop()
        finally:
            try:
                self.camera.release()
            except Exception:
                pass


def _setup_button(robot: Robot) -> None:
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: sem botão físico.")
        return
    try:
        for pin in CANDIDATE_BUTTON_PINS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            last_state = GPIO.input(pin)
            last_change = time.time()

            def _toggle(channel):
                nonlocal last_state, last_change
                try:
                    current_state = GPIO.input(channel)
                    current_time = time.time()
                    if current_state == GPIO.LOW and last_state == GPIO.HIGH and (current_time - last_change) > 0.3:
                        if robot.running:
                            robot.stop()
                        else:
                            robot.start()
                        last_change = current_time
                    last_state = current_state
                except Exception as e:
                    log(f"Erro no callback do botão BCM {channel}: {e}")

            GPIO.add_event_detect(pin, GPIO.BOTH, callback=_toggle, bouncetime=300)
            log(f"Botão físico pronto no BCM {pin}.")
            break
    except Exception as e:
        log(f"GPIO não funcional neste ambiente: {e}")
        return


def main():
    try:
        with Robot() as robot:
            _setup_button(robot)
            robot.start()
            try:
                log("Loop principal (pressione Ctrl+C para sair).")
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                log("Interrupção recebida.")
                robot.stop()
    except Exception as e:
        log(f"Erro fatal na inicialização: {e}")
        raise


if __name__ == "__main__":
    main()
