# main.py
# Segue-linha + UI web + botão físico (BCM21 ou BCM4) para START/STOP.
# Mantém STBY como está no hardware_control (você disse que já fica em HIGH).

import os
import cv2
import time
import json
import threading
import numpy as np
from datetime import datetime
from collections import deque

# ==== Estado WEB (compartilhado) ====
WEB_AVAILABLE = False
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

# ==== GPIO (opcional) ====
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

# Tentamos botão em BCM21; se não der, tentamos BCM4 (muito comum)
CANDIDATE_BUTTON_PINS = [21, 4]

from hardware_control import HardwareControl
from vision import Vision
from led_control import LedController

# ==== Picamera2 (opcional) ====
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


class Camera:
    """Picamera2 ou USB; entrega BGR; aplica rotate_180 se necessário."""
    def __init__(self, width=640, height=480, rotate_180=True):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                if Transform is not None:
                    cfg = self.picam.create_preview_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"},
                        transform=Transform(hflip=0, vflip=0),
                    )
                else:
                    cfg = self.picam.create_preview_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"}
                    )
                self.picam.configure(cfg)
                self.picam.start()
                log("Picamera2 iniciada.")
            except Exception as e:
                log(f"Falha Picamera2: {e}. Usando OpenCV/USB.")
                self.picam = None

        if self.picam is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
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

    BASE_SPEED = 55
    MIX_ANGLE = 0.7
    MAX_ANGLE = 50.0

    N_STRIPS = 8
    STRIP_H = 22
    STRIP_BOTTOM = 440

    BIN_BLUR = 3
    ADAPT_BLOCK = 21
    ADAPT_C = 7
    MORPH = 3

    SIDE_MARGIN_FRACTION = 0.15  # 15%

    INTERSECTION_WIDTH_PX = 180
    INTERSECT_DEBOUNCE = 2
    INTERSECT_AHEAD_DEBOUNCE = 2
    GREEN_DEBOUNCE = 2

    MAX_GAP_FRAMES = 8
    LINE_LOSS_GRACE_FRAMES = 12

    PID_DEFAULTS = {"kp": 0.9, "ki": 0.0, "kd": 0.14, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=True)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        # LEDs
        try:
            self.leds = LedController(pin=12, brightness=150)  # WS2812 (DIN em GPIO12)
            if self.leds and self.leds.enabled:
                self.leds.status_ok_idle()
        except Exception as e:
            self.leds = None
            log(f"LEDs desabilitados: {e}")

        self.history = deque(maxlen=5)
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

    # ===== ciclo de vida =====
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
        try:
            self.hardware.stop()
        except Exception:
            pass
        if getattr(self, "leds", None):
            try:
                self.leds.status_lost()
                self.leds.status_ok_idle()
            except Exception:
                pass
        log("Parado.")

    def cleanup(self):
        self.running = False
        try:
            self.hardware.cleanup()
        except Exception:
            pass
        self.camera.release()
        if getattr(self, "leds", None):
            try: self.leds.cleanup()
            except Exception: pass
        log("Recursos liberados.")

    # ===== UI =====
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
                try:
                    if hasattr(self.hardware, "config") and isinstance(self.hardware.config, dict):
                        self.hardware.config.update(cfg)
                    if hasattr(self.hardware, "update_pid_from_config"):
                        self.hardware.update_pid_from_config()
                except Exception as e:
                    log(f"Aviso: falha ao aplicar PID do config.json: {e}")
                log("Config carregada.")
            except Exception as e:
                log(f"Falha ao ler config.json: {e}")

    # ===== motores =====
    def _drive(self, base_speed: float, error: float):
        try:
            self.hardware.set_motor_speed(base_speed, error)
        except TypeError:
            left = int(max(-100, min(100, base_speed - error)))
            right = int(max(-100, min(100, base_speed + error)))
            self.hardware.set_motor_speed(left, right)

    # ===== visão =====
    def _binarize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.BIN_BLUR > 1:
            gray = cv2.GaussianBlur(gray, (self.BIN_BLUR, self.BIN_BLUR), 0)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            self.ADAPT_BLOCK, self.ADAPT_C
        )
        if self.MORPH > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MORPH, self.MORPH))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw[:, :self.LEFT_CROP] = 0
        bw[:, self.RIGHT_CROP:] = 0
        return bw

    def _strip_centroid(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, self.LEFT_CROP:self.RIGHT_CROP]
        colsum = roi.sum(axis=0)
        if colsum.max() < 255 * h * 0.05:
            return None, 0
        x = np.arange(self.LEFT_CROP, self.RIGHT_CROP, dtype=np.float32)
        cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
        cy = int((y0 + y1) / 2)
        center_band = roi[max(0, h//2-1):min(h, h//2+1), :]
        width_est = int(np.count_nonzero(center_band))
        return (int(cx), int(cy)), width_est

    def _fit_angle(self, pts):
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        alpha, beta = np.linalg.lstsq(A, xs, rcond=None)[0]
        angle_deg = np.degrees(np.arctan2(alpha, 1.0))
        return float(np.clip(angle_deg, -self.MAX_ANGLE, self.MAX_ANGLE))

    def _strip_profile(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, self.LEFT_CROP:self.RIGHT_CROP]
        colsum = roi.sum(axis=0).astype(np.float32)
        if colsum.max() > 0:
            colsum /= (255.0 * h)
        return colsum

    def _is_bimodal(self, profile, min_sep_px=60, min_peak=0.15):
        p = profile
        if p is None or len(p) < 5:
            return False
        k = max(5, (len(p)//100)*2+1)
        ker = np.ones(k, np.float32)/k
        ps = np.convolve(p, ker, mode="same")
        peaks = []
        for i in range(1, len(ps)-1):
            if ps[i] > ps[i-1] and ps[i] > ps[i+1] and ps[i] >= min_peak:
                peaks.append(i)
        if len(peaks) < 2:
            return False
        sep = max(peaks[j]-peaks[i] for i in range(len(peaks)) for j in range(i+1, len(peaks)))
        return sep >= min_sep_px

    def _detect_intersection_core(self, bw, widths, cents, angle_deg, idx_bottom=0, idx_mid=2):
        yb = self.STRIP_BOTTOM - idx_bottom*self.STRIP_H
        ym = self.STRIP_BOTTOM - idx_mid*self.STRIP_H
        prof_bottom = self._strip_profile(bw, yb, self.STRIP_H)
        prof_mid = self._strip_profile(bw, ym, self.STRIP_H)

        bimodal_bottom = self._is_bimodal(prof_bottom, min_sep_px=70, min_peak=0.16)
        bimodal_mid = self._is_bimodal(prof_mid, min_sep_px=60, min_peak=0.14)

        w_bottom = widths[idx_bottom] if idx_bottom < len(widths) else 0
        w_mid = widths[idx_mid] if idx_mid < len(widths) else 0
        wide_bottom = w_bottom >= self.INTERSECTION_WIDTH_PX
        wide_mid = w_mid >= self.INTERSECTION_WIDTH_PX * 0.85

        dx = 0.0
        up = idx_mid
        if up < len(cents) and cents[idx_bottom] and cents[up]:
            dx = float(cents[idx_bottom][0] - cents[up][0])
        big_shift = abs(dx) >= 90
        ang_high = abs(angle_deg) >= 28.0

        is_intersection = (bimodal_bottom or bimodal_mid) or (wide_bottom and wide_mid and not (ang_high and big_shift))
        is_curve90 = (ang_high and big_shift) and not (bimodal_bottom or bimodal_mid)
        return is_intersection, is_curve90

    def _detect_intersection(self, bw, widths, cents, angle_deg):
        is_now, is_c90_now = self._detect_intersection_core(bw, widths, cents, angle_deg, idx_bottom=0, idx_mid=2)
        top = self.N_STRIPS - 1
        mid_top = max(0, top - 2)
        is_ahead, _ = self._detect_intersection_core(bw, widths, cents, angle_deg, idx_bottom=mid_top, idx_mid=top)
        return is_now, is_c90_now, is_ahead

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
                is_intersection, is_curve90, is_ahead = self._detect_intersection(bw, widths, cents, angle)

                if is_ahead:
                    self._intersect_ahead_seen = min(self._intersect_ahead_seen + 1, 10)
                else:
                    self._intersect_ahead_seen = 0
                confirmed_ahead = self._intersect_ahead_seen >= self.INTERSECT_AHEAD_DEBOUNCE
                if getattr(self, "leds", None):
                    try:
                        if confirmed_ahead: self.leds.status_ahead()
                        else:               self.leds.status_ok()
                    except Exception:
                        pass

                c0 = cents[0]
                if c0 is None:
                    upper = [p for p in cents[1:] if p is not None]
                    if upper and (self._gap_frames_left < self.MAX_GAP_FRAMES):
                        steps = 1
                        est_x = (upper[0][0] + np.tan(np.radians(angle)) * steps * self.STRIP_H) if angle is not None else (self.history[-1] if self.history else self.WIDTH//2)
                        est_x = int(np.clip(est_x, self.LEFT_CROP, self.RIGHT_CROP-1))
                        c0 = (est_x, int(self.STRIP_BOTTOM + self.STRIP_H//2))
                        self._gap_frames_left += 1
                        self._line_loss_grace = 0
                    else:
                        if self._line_loss_grace < self.LINE_LOSS_GRACE_FRAMES:
                            self._line_loss_grace += 1
                            self._drive(self.BASE_SPEED, 0.0)
                            out = frame if SHARED_STATE.get("view_mode") != "preview" else self._draw_preview(
                                frame, bw, cents, angle, None, False, False, confirmed_ahead, None
                            )
                            self._publish(out, f"Linha sumiu — seguindo reto ({self._line_loss_grace}/{self.LINE_LOSS_GRACE_FRAMES})")
                            time.sleep(0.003)
                            continue
                        self._gap_frames_left = 0
                        self._line_loss_grace = 0
                        if getattr(self, "leds", None):
                            try:
                                self.leds.status_lost()
                                self.leds.status_ok_idle()
                            except Exception:
                                pass
                        self._publish(frame, "Linha perdida")
                        try: self.hardware.stop()
                        except Exception: pass
                        time.sleep(0.01)
                        continue
                else:
                    self._gap_frames_left = 0
                    self._line_loss_grace = 0

                self.history.append(c0[0])

                steps = 5
                dx = np.tan(np.radians(angle)) * steps * self.STRIP_H
                look = (c0[0] + dx, c0[1] - steps*self.STRIP_H)
                look = (int(np.clip(look[0], 0, self.WIDTH-1)),
                        int(np.clip(look[1], 0, self.HEIGHT-1)))

                if is_intersection:
                    self._intersect_seen = min(self._intersect_seen + 1, 10)
                else:
                    self._intersect_seen = 0
                confirmed_intersection = self._intersect_seen >= self.INTERSECT_DEBOUNCE
                if confirmed_intersection and getattr(self, "leds", None):
                    try: self.leds.status_intersection()
                    except Exception: pass

                green_centroids, green_dir = self.vision.detect_greens(frame)
                if green_dir:
                    self._green_last = green_dir
                    self._green_seen = min(self._green_seen + 1, 10)
                else:
                    self._green_seen = 0
                confirmed_green = self._green_last if self._green_seen >= self.GREEN_DEBOUNCE else None

                now = time.time()
                if confirmed_intersection and not is_curve90:
                    if confirmed_green == "uturn":
                        log("UTURN: dois verdes — retornando até reacoplar.")
                        if getattr(self, "leds", None):
                            try: self.leds.status_turn("uturn")
                            except Exception: pass
                        t0 = time.time()
                        timeout = 3.0
                        while time.time() - t0 < timeout and self.running:
                            try:
                                self.hardware.set_motor_speed(0, 120)
                            except TypeError:
                                self.hardware.set_motor_speed(-60, 60)
                            re_bw = self._binarize(self.camera.read())
                            c_re, w0_re = self._strip_centroid(re_bw, self.STRIP_BOTTOM, self.STRIP_H)
                            if w0_re > 40 and c_re is not None:
                                break
                        self.hardware.stop()
                        self._green_seen = 0
                        self._intersect_seen = 0
                        self._green_last = None
                        self.planned_direction = None
                        continue
                    elif confirmed_green in ("left", "right"):
                        self.planned_direction = confirmed_green
                        self.turning_until = now + 0.7
                        if getattr(self, "leds", None):
                            try: self.leds.status_turn(confirmed_green)
                            except Exception: pass
                        log(f"Curva {confirmed_green} marcada — viés por 0.7s.")
                        self._green_seen = 0
                        self._intersect_seen = 0
                    else:
                        self.planned_direction = "straight"
                        self.turning_until = now + 0.4
                        if getattr(self, "leds", None):
                            try: self.leds.status_following()
                            except Exception: pass
                        log("Interseção sem marca — seguindo reto.")

                offset = c0[0] - (self.WIDTH // 2)
                error = float(offset) + self.MIX_ANGLE * float(angle)

                if self.planned_direction and now < self.turning_until:
                    bias = 120 if self.planned_direction == "left" else (-120 if self.planned_direction == "right" else 0)
                    error += bias
                elif self.planned_direction and now >= self.turning_until:
                    self.planned_direction = None

                self._drive(self.BASE_SPEED, error)

                out = frame
                if SHARED_STATE.get("view_mode") == "preview":
                    out = self._draw_preview(frame, bw, cents, angle, look, confirmed_intersection, is_curve90, confirmed_ahead, confirmed_green)

                self._publish(out, "OK")
                time.sleep(0.003)

        except KeyboardInterrupt:
            log("Interrompido.")
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try: self.hardware.stop()
            except Exception: pass

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
            log(f"Status: {status_msg} | FPS ~ {fps:.1f} | L/R: {left}/{right}")


# ==== WEB / comandos ====
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


# ==== Botão físico (tenta BCM21 e BCM4) ====
def _setup_button(robot: Robot):
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: iniciando sem botão físico.")
        return

    configured = False
    chosen_pin = None

    for pin in CANDIDATE_BUTTON_PINS:
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            # limpa eventos prévios (se já havia teste)
            try:
                GPIO.remove_event_detect(pin)
            except Exception:
                pass

            def _toggle(channel):
                try:
                    # debounce por software
                    time.sleep(0.03)
                    if GPIO.input(pin) == GPIO.LOW:
                        if robot.running:
                            log(f"Botão (BCM {pin}): STOP")
                            robot.stop()
                        else:
                            log(f"Botão (BCM {pin}): START")
                            robot.start()
                except Exception as e:
                    log(f"Erro no botão (BCM {pin}): {e}")

            GPIO.add_event_detect(pin, GPIO.FALLING, callback=_toggle, bouncetime=300)
            configured = True
            chosen_pin = pin
            log(f"Botão físico pronto no BCM {pin} (pull-up, FALLING).")
            break
        except Exception as e:
            log(f"Falha ao configurar botão no BCM {pin}: {e}")

    if not configured:
        log("Nenhum botão configurado (BCM21/4). Siga pela UI/web.")
        return

    # Polling de segurança: se o usuário mantiver pressionado na hora errada, ainda detectamos
    def _safety_poll():
        last_state = GPIO.input(chosen_pin)
        while True:
            try:
                state = GPIO.input(chosen_pin)
                if state != last_state:
                    last_state = state
                    if state == GPIO.LOW:
                        # espelha a lógica do callback
                        if robot.running:
                            log(f"[poll] Botão (BCM {chosen_pin}): STOP")
                            robot.stop()
                        else:
                            log(f"[poll] Botão (BCM {chosen_pin}): START")
                            robot.start()
                time.sleep(0.05)
            except Exception:
                break

    th = threading.Thread(target=_safety_poll, daemon=True)
    th.start()


def main():
    robot = Robot()
    _wire_web(robot)
    _setup_button(robot)

    if WEB_AVAILABLE and hasattr(web_stream, "app"):
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5000"))
        log(f"Servidor web em http://{host}:{port}")

        # Não inicia automaticamente: você usa o botão físico ou o botão da UI.
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
