# main.py
# Segue-linha robusto + UI web + botão físico (21/4).
# Câmera rotacionada 180°. LEDs WS2812 desligados por padrão (USE_LEDS=False).
# Usa HardwareControl (lgpio) e Vision (greens).

import os
import cv2
import time
import json
import threading
import numpy as np
from datetime import datetime
from collections import deque

# ===== LEDs OFF por segurança (evitar travar PWM) =====
USE_LEDS = False

# ==== Estado WEB ====
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

# ==== GPIO opcional (só pro botão) ====
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

CANDIDATE_BUTTON_PINS = [21, 4]

from hardware_control import HardwareControl
from vision import Vision
try:
    from led_control import LedController
except Exception:
    LedController = None

# ==== Picamera2 opcional ====
PICAMERA_AVAILABLE = False
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

# ------------------ Câmera (com rotate 180°) ------------------
class Camera:
    """Picamera2 em modo vídeo; retorno BGR. Sem rotação (imagem em pé)."""
    def __init__(self, width=640, height=480, rotate_180=False):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None

        # Dica: OpenCV usa todos os núcleos por padrão; no Pi isso compete com o encoder
        try:
            import cv2 as _cv2
            _cv2.setNumThreads(1)
        except Exception:
            pass

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                # Modo de VÍDEO (menor latência que preview), menos buffers
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

                # Força ~30 fps se o sensor permitir (limites em microssegundos)
                # 1/30s ~= 33333us. (10_000, 33_333) = 30–100 fps aprox
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
            # Fallback USB (mantém como estava)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Nenhuma câmera disponível.")

    def read(self):
        if self.picam is not None:
            rgb = self.picam.capture_array()  # zero-copy do pipeline
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

# ------------------ Robô ------------------
class Robot:
    WIDTH = 640
    HEIGHT = 480

    LINE_IS_DARK = True
    CREEP_WHEN_LOST = True
    CREEP_SPEED = 45  # Faster creep for quicker recovery
    BASE_SPEED = 30
    MIX_ANGLE = 0.5  # Less angle weight for stability in curves
    MAX_ANGLE = 50.0

    N_STRIPS = 6  # Fewer strips for tight curves
    STRIP_H = 28  # Taller strips for better centroid stability
    STRIP_BOTTOM = 460  # Lower for near-field focus

    BIN_BLUR = 3
    ADAPT_BLOCK = 21
    ADAPT_C = 7
    MORPH = 3

    SIDE_MARGIN_FRACTION = 0.08  # Wider view for curves

    INTERSECTION_WIDTH_PX = 220
    INTERSECTION_WIDTH_FRAC = 0.55
    INTERSECT_DEBOUNCE = 4
    INTERSECT_AHEAD_DEBOUNCE = 3

    GREEN_DEBOUNCE = 2
    MAX_GAP_FRAMES = 8
    LINE_LOSS_GRACE_FRAMES = 18  # More grace to avoid stopping

    PID_DEFAULTS = {"kp": 0.65, "ki": 0.02, "kd": 0.18, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=False)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        self.leds = None
        if USE_LEDS and LedController is not None:
            try:
                self.leds = LedController(pin=12, brightness=150)
                if self.leds and self.leds.enabled:
                    self.leds.status_ok_idle()
            except Exception as e:
                self.leds = None
                log(f"LEDs desabilitados: {e}")

        self.history = deque(maxlen=5)
        self.angle_history = deque(maxlen=3)  # For angle smoothing
        self._intersect_seen = 0
        self._intersect_ahead_seen = 0
        self._green_seen = 0
        self._green_last = None
        self.planned_direction = None
        self.turning_until = 0.0
        self._gap_frames_left = 0
        self._line_loss_grace = 0

        self.cfg_path = "config.json"
        self._load_config_if_any()

        self._last_ts = time.time()
        self._frames = 0

        self.LEFT_CROP = int(self.WIDTH * self.SIDE_MARGIN_FRACTION)
        self.RIGHT_CROP = self.WIDTH - int(self.WIDTH * self.SIDE_MARGIN_FRACTION)

    # ---- ciclo de vida ----
    def start(self):
        if self.running:
            log("Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        SHARED_STATE["status"] = "running"
        if getattr(self, "leds", None):
            try: self.leds.status_ok()
            except Exception: pass
        log("Loop principal iniciado.")

    def stop(self):
        self.running = False
        SHARED_STATE["status"] = "stopped"
        try: self.hardware.stop()
        except Exception: pass
        if getattr(self, "leds", None):
            try:
                self.leds.status_lost()
                self.leds.status_ok_idle()
            except Exception: pass
        log("Parado.")

    def cleanup(self):
        self.running = False
        try: self.hardware.cleanup()
        except Exception: pass
        self.camera.release()
        if getattr(self, "leds", None):
            try: self.leds.cleanup()
            except Exception: pass
        log("Recursos liberados.")

    # ---- UI ----
    def set_view_mode(self, mode: str):
        SHARED_STATE["view_mode"] = mode
        log(f"View mode -> {mode}")

    def calibrate_pixel(self, x, y, color: str):
        frame = SHARED_STATE.get("last_frame", None)
        if frame is None:
            log("Sem frame para calibrar.")
            return False
        x = max(0, min(frame.shape[1] - 1, int(x)))
        y = max(0, min(frame.shape[0] - 1, int(y)))
        ok = self.vision.calibrate_by_click(frame, x, y, color)
        log(f"Calibração ({color}) em ({x},{y}) -> {ok}")
        return ok

    def save_config(self, new_cfg: dict):
        try:
            SHARED_STATE["config"].update(new_cfg or {})
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                json.dump(SHARED_STATE["config"], f, ensure_ascii=False, indent=2)
            if "vision" in SHARED_STATE["config"]:
                self.vision.update_config(SHARED_STATE["config"]["vision"])
            try:
                if hasattr(self.hardware, "config") and isinstance(self.hardware.config, dict):
                    self.hardware.config.update(SHARED_STATE["config"])
                if hasattr(self.hardware, "update_pid_from_config"):
                    self.hardware.update_pid_from_config()
            except Exception as e:
                log(f"Aviso: falha ao propagar PID: {e}")
            log("Config salva.")
            return True
        except Exception as e:
            log(f"Falha save_config: {e}")
            return False

    def _load_config_if_any(self):
        if os.path.exists(self.cfg_path):
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                SHARED_STATE["config"] = cfg
                if "vision" in cfg:
                    self.vision.update_config(cfg["vision"])
                if "pid" in cfg:
                    self.hardware.config.update({"pid": cfg["pid"]})
                    if hasattr(self.hardware, "update_pid_from_config"):
                        self.hardware.update_pid_from_config()
                log("Config carregada.")
            except Exception as e:
                log(f"Falha ao carregar config: {e}")

    # ---- Visão ----
    def _binarize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.BIN_BLUR > 0:
            gray = cv2.GaussianBlur(gray, (self.BIN_BLUR * 2 + 1, self.BIN_BLUR * 2 + 1), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.ADAPT_BLOCK, self.ADAPT_C
        ) if self.LINE_IS_DARK else cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.ADAPT_BLOCK, self.ADAPT_C
        )
        if self.MORPH > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MORPH, self.MORPH))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)
        thresh[:, :self.LEFT_CROP] = 0
        thresh[:, self.RIGHT_CROP:] = 0
        return thresh

    def _strip_centroid(self, binary, y_base, h):
        y0 = int(y_base - h)
        y1 = int(y_base)
        strip = binary[y0:y1, self.LEFT_CROP:self.RIGHT_CROP]
        M = cv2.moments(strip)
        if M["m00"] == 0:
            return None, 0
        cx = M["m10"] / M["m00"] + self.LEFT_CROP
        cy = M["m01"] / M["m00"] + y0
        return (cx, cy), int(M["m00"] / 255)

    def _fit_angle(self, centroids):
        valids = [c for c in centroids if c is not None]
        if len(valids) < 2:
            return 0.0
        points = np.array(valids, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy, vx) * 180.0 / np.pi
        angle = -angle
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        return max(-self.MAX_ANGLE, min(self.MAX_ANGLE, angle))

    def _drive(self, base_speed, error):
        if base_speed == 0:
            self.hardware.stop()
        else:
            self.hardware.set_motor_speed(base_speed, error)

    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()
                bw = self._binarize(frame)
                cents = []
                w_sum = 0
                for i in range(self.N_STRIPS):
                    c, w = self._strip_centroid(bw, self.STRIP_BOTTOM - i * self.STRIP_H, self.STRIP_H)
                    cents.append(c)
                    w_sum += w
                c0 = cents[0]
                valids = [c for c in cents if c is not None]
                fit_angle = self._fit_angle(cents)
                self.angle_history.append(fit_angle)
                fit_angle = np.mean(self.angle_history) if self.angle_history else fit_angle

                look_point = None
                if len(valids) >= 2:
                    look = valids[-1]
                    look_point = look
                    self.history.append(look[0])
                else:
                    self.history.append(None)

                is_intersection = False
                is_ahead = False
                if c0 is not None:
                    is_intersection = w_sum > self.INTERSECTION_WIDTH_PX
                    if is_intersection:
                        self._intersect_seen += 1
                        if self._intersect_seen < self.INTERSECT_DEBOUNCE:
                            is_intersection = False
                    else:
                        self._intersect_seen = 0
                    if not is_intersection:
                        ahead_cent, ahead_w = self._strip_centroid(bw, self.STRIP_BOTTOM - self.N_STRIPS * self.STRIP_H, self.STRIP_H)
                        is_ahead = ahead_w > self.INTERSECTION_WIDTH_PX * self.INTERSECTION_WIDTH_FRAC
                        if is_ahead:
                            self._intersect_ahead_seen += 1
                            if self._intersect_ahead_seen < self.INTERSECT_AHEAD_DEBOUNCE:
                                is_ahead = False
                        else:
                            self._intersect_ahead_seen = 0
                else:
                    self._intersect_seen = 0
                    self._intersect_ahead_seen = 0

                is_curve90 = len(valids) <= 4 and abs(fit_angle) > 25 and not is_intersection

                greens, green_dir = self.vision.detect_greens(frame)
                confirmed_green = None
                if green_dir is not None:
                    self._green_seen += 1
                    if self._green_seen >= self.GREEN_DEBOUNCE:
                        confirmed_green = green_dir
                        self._green_last = green_dir
                else:
                    self._green_seen = 0

                if confirmed_green == "uturn":
                    self._gap_frames_left = self.MAX_GAP_FRAMES
                    self._line_loss_grace = self.LINE_LOSS_GRACE_FRAMES

                now = time.time()
                if c0 is None:
                    self._line_loss_grace -= 1
                    if self._line_loss_grace <= 0 and self.CREEP_WHEN_LOST:
                        if self._gap_frames_left > 0:
                            self._gap_frames_left -= 1
                            self._drive(self.CREEP_SPEED, 0)
                        elif self._green_last is not None and self._green_last in ("left", "right"):
                            timeout = 0.8
                            t0 = time.time()
                            while time.time() - t0 < timeout and self.running:
                                try:
                                    self.hardware.set_motor_speed(0, 120 if self._green_last == "left" else -120)
                                except TypeError:
                                    self.hardware.set_motor_speed(-60, 60 if self._green_last == "left" else -60)
                                re_bw = self._binarize(self.camera.read())
                                c_re, w0_re = self._strip_centroid(re_bw, self.STRIP_BOTTOM, self.STRIP_H)
                                if w0_re > 40 and c_re is not None:
                                    break
                            self.hardware.stop()
                            self._green_seen = 0; self._intersect_seen = 0; self._green_last = None
                            self.planned_direction = None
                            continue
                    elif self._line_loss_grace <= 0:
                        self.hardware.stop()
                        self._publish(frame, "NO_LINE")
                        time.sleep(0.01)
                        continue
                    else:
                        self._drive(self.CREEP_SPEED, 0)
                else:
                    self._line_loss_grace = self.LINE_LOSS_GRACE_FRAMES
                    self._gap_frames_left = max(0, self._gap_frames_left - 1)

                if is_intersection or is_ahead:
                    self._green_seen = 0; self._green_last = None
                    self.planned_direction = None
                    self._drive(self.CREEP_SPEED, 0)
                    self._publish(frame, "INTERSECTION")
                    time.sleep(0.01)
                    continue
                elif is_curve90:
                    bias = np.sign(fit_angle) * 250
                    error = float(c0[0] - (self.WIDTH // 2)) + self.MIX_ANGLE * float(fit_angle) + bias
                    self._drive(self.BASE_SPEED, error)
                    self._publish(frame, "CURVE90")
                elif confirmed_green in ("left", "right"):
                    self.planned_direction = confirmed_green
                    self.turning_until = now + 0.7
                    if getattr(self, "leds", None):
                        try: self.leds.status_turn(confirmed_green)
                        except Exception: pass
                    self._green_seen = 0; self._intersect_seen = 0
                else:
                    self.planned_direction = "straight"
                    self.turning_until = now + 0.4
                    if getattr(self, "leds", None):
                        try: self.leds.status_following()
                        except Exception: pass

                offset = c0[0] - (self.WIDTH // 2)
                error = float(offset) + self.MIX_ANGLE * float(fit_angle)

                if self.planned_direction and now < self.turning_until:
                    bias = 120 if self.planned_direction == "left" else (-120 if self.planned_direction == "right" else 0)
                    error += bias
                elif self.planned_direction and now >= self.turning_until:
                    self.planned_direction = None

                self._drive(self.BASE_SPEED, error)

                out = frame
                if SHARED_STATE.get("view_mode") == "preview":
                    out = self._draw_preview(frame, bw, cents, fit_angle, look_point, is_intersection, is_curve90, is_ahead, confirmed_green)

                self._publish(out, "OK")
                time.sleep(0.003)

        except KeyboardInterrupt:
            log("Interrompido.")
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try: self.hardware.stop()
            except Exception: pass

    # ---- helpers de preview/publicação ----
    def _draw_preview(self, frame, bw, cents, fit_angle, look_point, is_intersection, is_curve90, is_ahead, green_dir):
        out = frame.copy()
        y_base = int(self.STRIP_BOTTOM)
        cv2.line(out, (0, y_base), (self.WIDTH-1, y_base), (0, 0, 255), 2)
        cv2.rectangle(out, (0, 0), (self.LEFT_CROP, self.HEIGHT-1), (40, 40, 40), 1)
        cv2.rectangle(out, (self.RIGHT_CROP, 0), (self.WIDTH-1, self.HEIGHT-1), (40, 40, 40), 1)
        for i, c in enumerate(cents):
            y0 = int(self.STRIP_BOTTOM - i*self.STRIP_H)
            cv2.rectangle(out, (self.LEFT_CROP, y0), (self.RIGHT_CROP-1, y0 + self.STRIP_H), (80, 80, 80), 1)
            if c is not None:
                cv2.circle(out, (int(c[0]), int(c[1])), 6, (0, 200, 0), -1, cv2.LINE_AA)
        valids = [c for c in cents if c is not None]
        if len(valids) >= 2:
            pA, pB = valids[0], valids[-1]
            cv2.line(out, (int(pA[0]), int(pA[1])), (int(pB[0]), int(pB[1])), (255, 0, 0), 2, cv2.LINE_AA)
        if look_point is not None:
            cv2.circle(out, (int(look_point[0]), int(look_point[1])), 8, (0, 255, 255), 2, cv2.LINE_AA)
        txt = f"angle={fit_angle:+.1f} strips={len(valids)}"
        if is_intersection: txt += "  INT"
        if is_curve90: txt += "  C90"
        if is_ahead: txt += "  AHEAD"
        if green_dir: txt += f"  GREEN:{green_dir}"
        cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return out

    def _publish(self, frame, status_msg):
        left = getattr(self.hardware, "last_left_speed", 0)
        right = getattr(self.hardware, "last_right_speed", 0)
        SHARED_STATE["last_frame"] = frame
        SHARED_STATE["speeds"] = {"left": left, "right": right}
        SHARED_STATE["status"] = status_msg
        self._frames += 1
        now = time.time()
        if now - getattr(self, "_last_ts", now) >= 2.0:
            fps = self._frames / (now - self._last_ts)
            self._frames = 0
            self._last_ts = now
            SHARED_STATE["fps"] = round(fps, 1)
            valids = [c for c in SHARED_STATE["last_frame"] if c is not None]
            log(f"Status: {status_msg} | FPS ~ {fps:.1f} | L/R: {left}/{right} | Centroids: {len(valids)} | Angle: {getattr(self, 'fit_angle', 0):.1f} | Error: {getattr(self, 'error', 0):.1f} | Curve90: {getattr(self, 'is_curve90', False)}")

# ------------------ Web/Socket ------------------
def _wire_web(robot: Robot):
    if not WEB_AVAILABLE:
        return
    try:
        if hasattr(web_stream, "register_robot"):
            web_stream.register_robot(robot)
            log("Robô registrado no servidor web.")
        elif hasattr(web_stream, "socketio"):
            sio = web_stream.socketio
            @sio.on("command")
            def on_command(cmd):
                try:
                    name = cmd.get("name")
                    data = cmd.get("data", {})
                    if name == "start_robot":
                        robot.start()
                    elif name == "stop_robot":
                        robot.stop()
                    elif name == "set_view_mode":
                        robot.set_view_mode(data.get("mode", "preview"))
                    elif name == "calibrate_pixel":
                        robot.calibrate_pixel(int(data["x"]), int(data["y"]), data.get("color", "green"))
                    elif name == "save_config":
                        robot.save_config(data or {})
                    else:
                        log(f"Comando desconhecido: {cmd}")
                except Exception as e:
                    log(f"Erro no comando via socket: {e}")
        else:
            log("web_stream sem app/socketio/register_robot; headless.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")

# ------------------ Botão físico ------------------
def _setup_button(robot: Robot):
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: iniciando sem botão físico.")
        return
    configured = False
    chosen_pin = None
    for pin in [21, 4]:
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            try: GPIO.remove_event_detect(pin)
            except Exception: pass
            def _toggle(channel):
                try:
                    time.sleep(0.03)
                    if GPIO.input(pin) == GPIO.LOW:
                        if robot.running:
                            log(f"Botão (BCM {pin}): STOP"); robot.stop()
                        else:
                            log(f"Botão (BCM {pin}): START"); robot.start()
                except Exception as e:
                    log(f"Erro no botão (BCM {pin}): {e}")
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=_toggle, bouncetime=300)
            configured = True; chosen_pin = pin
            log(f"Botão físico pronto no BCM {pin} (pull-up, FALLING).")
            break
        except Exception as e:
            log(f"Falha ao configurar botão no BCM {pin}: {e}")
    if not configured:
        log("Nenhum botão configurado (BCM21/4). Siga pela UI/web.")
        return
    def _safety_poll():
        last_state = GPIO.input(chosen_pin)
        while True:
            try:
                state = GPIO.input(chosen_pin)
                if state != last_state:
                    last_state = state
                    if state == GPIO.LOW:
                        if robot.running:
                            log(f"[poll] Botão (BCM {chosen_pin}): STOP"); robot.stop()
                        else:
                            log(f"[poll] Botão (BCM {chosen_pin}): START"); robot.start()
                time.sleep(0.05)
            except Exception:
                break
    th = threading.Thread(target=_safety_poll, daemon=True)
    th.start()

# ------------------ main ------------------
def main():
    robot = Robot()
    _wire_web(robot)
    _setup_button(robot)

    if WEB_AVAILABLE and hasattr(web_stream, "app"):
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5000"))
        log(f"Servidor web em http://{host}:{port}")
        log("Aguardando START (botão físico ou UI/web)...")
        try:
            if hasattr(web_stream, "socketio"):
                web_stream.socketio.run(web_stream.app, host=host, port=port, allow_unsafe_werkzeug=True)
            else:
                web_stream.app.run(host=host, port=port)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()
            if GPIO_AVAILABLE:
                try: GPIO.cleanup()
                except Exception: pass
    else:
        log("Rodando sem servidor web.")
        log("Use o botão físico (se configurado) para START/STOP.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()
            if GPIO_AVAILABLE:
                try: GPIO.cleanup()
                except Exception: pass

if __name__ == "__main__":
    main()
