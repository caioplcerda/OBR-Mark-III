import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple

USE_LEDS = True
WEB_AVAILABLE = True
GPIO_AVAILABLE = True

try:
    import web_stream
    if hasattr(web_stream, "SHARED_STATE"):
        WEB_AVAILABLE = True
except Exception:
    WEB_AVAILABLE = False

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
except Exception:
    PICAMERA_AVAILABLE = False


class SharedState:
    """Gerencia o estado compartilhado entre threads com bloqueio de concorrência."""
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

    def get(self, key: str) -> any:
        with self._lock:
            return self._state.get(key)

    def set(self, key: str, value: any) -> None:
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
    """Adiciona uma mensagem ao log com timestamp."""
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
    """Gerencia a captura de vídeo via Picamera2 ou OpenCV."""
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
                cfg = self.picam.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    transform=Transform(hflip=0, vflip=0) if Transform else None,
                    buffer_count=3
                )
                self.picam.configure(cfg)
                self.picam.set_controls({"FrameDurationLimits": (10000, 33333)})
                self.picam.start()
                log("Picamera2 iniciada em modo vídeo.")
            except Exception as e:
                log(f"Falha Picamera2: {e}. Usando OpenCV/USB.")
                self.picam = None

        if self.picam is None:
            self._init_usb_camera()

    def _init_usb_camera(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Nenhuma câmera disponível.")

    def _reconnect_usb(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
            self._init_usb_camera()
            log("Câmera USB reconectada com sucesso.")
        except Exception as e:
            log(f"Falha ao reconectar câmera USB: {e}")
            raise RuntimeError("Falha ao reconectar câmera USB.")

    def read(self) -> np.ndarray:
        for _ in range(3):
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
                log(f"Erro ao ler frame: {e}. Tentando novamente...")
                time.sleep(0.1)
        raise RuntimeError("Falha persistente na câmera.")

    def release(self) -> None:
        try:
            if self.picam is not None:
                self.picam.stop()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class LineFollowerController:
    """Controlador PD para seguir linha com suavização de erro."""
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
    """Controla um robô seguidor de linha com visão computacional."""
    def __init__(self):
        config = CONFIG_DEFAULTS
        self.camera: Camera = Camera(**config["camera"])
        self.vision: Vision = Vision({}, log)
        self.hardware: HardwareControl = HardwareControl({"pid": {}})
        self.controller: LineFollowerController = LineFollowerController(config)
        self.width: int = config["camera"]["width"]
        self.height: int = config["camera"]["height"]
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Inicia o loop principal do robô."""
        if self.running:
            log("Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        SHARED_STATE.set("status", "running")
        log("Loop principal iniciado.")

    def stop(self) -> None:
        """Para o robô e libera recursos."""
        self.running = False
        SHARED_STATE.set("status", "stopped")
        try:
            self.hardware.stop()
        except Exception:
            pass
        log("Parado.")

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
                self.hardware.drive(left_speed, right_speed)
                SHARED_STATE.set("last_frame", frame)
                SHARED_STATE.set("speeds", {
                    "left": left_speed,
                    "right": right_speed
                })
                elapsed = time.time() - start_time
                sleep_time = max(0, target_period - elapsed)
                time.sleep(sleep_time)
                SHARED_STATE.set("fps", 1.0 / (elapsed + sleep_time) if elapsed + sleep_time > 0 else 0)
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            self.stop()

    def _binarize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binariza a imagem para detecção de linha."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        return bw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.camera.release()


def _wire_web(robot: Robot) -> None:
    """Integra o robô com o servidor web, se disponível."""
    if not WEB_AVAILABLE:
        return
    try:
        if hasattr(web_stream, "register_robot"):
            web_stream.register_robot(robot)
            log("Robô registrado no servidor web.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")


def _setup_button(robot: Robot) -> None:
    """Configura botão físico para iniciar/parar o robô."""
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: sem botão físico.")
        return
    for pin in CANDIDATE_BUTTON_PINS:
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            last_state = GPIO.HIGH
            last_change = time.time()

            def _toggle(channel):
                nonlocal last_state, last_change
                current_state = GPIO.input(pin)
                current_time = time.time()
                if current_state == GPIO.LOW and last_state == GPIO.HIGH and (current_time - last_change) > 0.3:
                    if robot.running:
                        robot.stop()
                    else:
                        robot.start()
                    last_change = current_time
                last_state = current_state

            GPIO.add_event_detect(pin, GPIO.BOTH, callback=_toggle)
            log(f"Botão físico pronto no BCM {pin}.")
            break
        except Exception as e:
            log(f"Falha botão BCM {pin}: {e}")


def main():
    """Função principal para executar o robô."""
    with Robot() as robot:
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
