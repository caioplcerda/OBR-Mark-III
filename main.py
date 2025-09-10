# main.py (corrigido + logs e controle de curvas)
# - Usa tempo para movimentos (forward/turn)
# - Logs claros quando vê interseções, curvas 90 e marcador verde
# - Turn direction invert flag (caso esteja virando para o lado errado)
# - Parâmetros de curva e velocidade fáceis de mudar via config.json ou via comando socket `save_config`

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

    BASE_SPEED = 30
    MIX_ANGLE = 0.7
    MAX_ANGLE = 50.0

    N_STRIPS = 5
    STRIP_H = 28
    STRIP_BOTTOM = 440

    INTERSECT_DEBOUNCE = 6
    INTERSECT_AHEAD_DEBOUNCE = 4
    C90_DEBOUNCE = 6
    GREEN_DEBOUNCE = 2

    # TEMPOS EM SEGUNDOS (fáceis de ajustar)
    INTERSECT_FWD_TIME = 0.8
    TURN90_FWD_TIME = 0.4
    TURN90_TURN_TIME = 1.2

    # Força do giro (menor -> giro mais lento). Ajuste facilmente via config.
    TURN_BIAS = 80
    # Se o robô estiver virando para o lado invertido, coloque True
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

        self.cfg_path = "config.json"
        self._load_config_if_any()

        self._last_ts = time.time()
        self._frames = 0

        self.LEFT_CROP = int(self.WIDTH * 0.15)
        self.RIGHT_CROP = self.WIDTH - int(self.WIDTH * 0.15)

    # ----------------- Config -----------------
    def _load_config_if_any(self):
        if os.path.exists(self.cfg_path):
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                SHARED_STATE["config"] = cfg
                # Aplicar parâmetros de curva se presentes
                self.update_turn_params(cfg.get("turn_params", {}))
                if "vision" in cfg:
                    self.vision.update_config(cfg["vision"])
                log("Config carregada.")
            except Exception as e:
                log(f"Falha ao ler config.json: {e}")

    def save_config(self, new_cfg: dict):
        # Pode ser chamado via socket: { name: 'save_config', data: {...} }
        try:
            if not isinstance(new_cfg, dict):
                return False
            SHARED_STATE["config"].update(new_cfg or {})
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                json.dump(SHARED_STATE["config"], f, ensure_ascii=False, indent=2)
            # Atualiza parâmetros dinâmicos
            self.update_turn_params(SHARED_STATE["config"].get("turn_params", {}))
            if "vision" in SHARED_STATE["config"]:
                self.vision.update_config(SHARED_STATE["config"]["vision"])
            log("Config salva e aplicada.")
            return True
        except Exception as e:
            log(f"Falha save_config: {e}")
            return False

    def update_turn_params(self, params: dict):
        if not isinstance(params, dict):
            return
        tp = params
        if "turn_bias" in tp:
            try:
                self.TURN_BIAS = float(tp["turn_bias"])
            except Exception:
                pass
        if "turn_time" in tp:
            try:
                self.TURN90_TURN_TIME = float(tp["turn_time"])
            except Exception:
                pass
        if "turn_fwd_time" in tp:
            try:
                self.TURN90_FWD_TIME = float(tp["turn_fwd_time"])
            except Exception:
                pass
        if "intersect_fwd_time" in tp:
            try:
                self.INTERSECT_FWD_TIME = float(tp["intersect_fwd_time"])
            except Exception:
                pass
        if "invert_turn_direction" in tp:
            try:
                self.TURN_DIRECTION_INVERT = bool(tp["invert_turn_direction"])
            except Exception:
                pass
        log(f"Turn params atualizados: bias={self.TURN_BIAS} turn_time={self.TURN90_TURN_TIME} invert={self.TURN_DIRECTION_INVERT}")

    # ----------------- ciclo de vida -----------------
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

    # ----------------- movimentos por TEMPO -----------------
    def _forward_time(self, duration_s: float, reason: str = "forward"):
        log(f"Iniciando forward: {duration_s:.2f}s  reason={reason}")
        self.hardware.set_motor_speed(self.BASE_SPEED, 0)
        start_time = time.time()
        while self.running:
            if time.time() - start_time >= duration_s:
                break
            time.sleep(0.01)
        self.hardware.stop()
        log(f"Forward finalizado: reason={reason}")

    def _turn_in_place_time(self, direction: str, duration_s: float, reason: str = "turn"):
        # direction: 'left' or 'right'
        d = direction.lower()
        if d not in ("left", "right"):
            d = "left"
        # aplica invert flag
        if self.TURN_DIRECTION_INVERT:
            d = "left" if d == "right" else "right"
        bias = self.TURN_BIAS if d == "left" else -self.TURN_BIAS
        log(f"Iniciando turn: dir={direction} (aplicada={d}) duration={duration_s:.2f}s bias={bias} reason={reason}")
        start_time = time.time()
        while self.running:
            if time.time() - start_time >= duration_s:
                break
            # usar set_motor_speed(0, bias) segue a lógica já existente no código
            try:
                self.hardware.set_motor_speed(0, bias)
            except Exception:
                pass
            time.sleep(0.01)
        self.hardware.stop()
        log(f"Turn finalizado: dir={direction} (aplicada={d}) reason={reason}")

    # ----------------- visão e detecção -----------------
    def _binarize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
        bw[:, :self.LEFT_CROP] = 0
        bw[:, self.RIGHT_CROP:] = 0
        return bw

    def _strip_centroid(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, self.LEFT_CROP:self.RIGHT_CROP]
        if roi.size == 0:
            return None, 0
        colsum = roi.sum(axis=0)
        if colsum.max() < 255 * h * 0.05:
            return None, 0
        x = np.arange(self.LEFT_CROP, self.RIGHT_CROP, dtype=np.float32)
        cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
        cy = int((y0 + y1) / 2)
        width_est = int(np.count_nonzero(roi, axis=1).sum() / max(1, roi.shape[0]))
        return (int(cx), int(cy)), width_est

    def _fit_angle(self, pts):
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        alpha, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        angle_deg = np.degrees(np.arctan2(alpha, 1.0))
        return float(np.clip(angle_deg, -self.MAX_ANGLE, self.MAX_ANGLE))

    def _is_bimodal(self, profile, min_sep_px=80, min_peak=0.20):
        if profile is None or len(profile) < 7:
            return False
        k = max(7, (len(profile)//80)*2+1)
        ker = np.ones(k, np.float32)/k
        ps = np.convolve(profile, ker, mode="same")
        peaks = []
        for i in range(2, len(ps)-2):
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
        wide_bottom = w_bottom >= 220
        wide_mid = w_mid >= 220 * 0.85
        dx = 0.0
        up = idx_mid
        if up < len(cents) and cents[idx_bottom] and cents[up]:
            dx = float(cents[idx_bottom][0] - cents[up][0])
        big_shift = abs(dx) >= 90
        ang_high = abs(angle_deg) >= 28.0
        is_intersection_raw = (bimodal_bottom or bimodal_mid) or (wide_bottom and wide_mid and not (ang_high and big_shift))
        is_curve90_raw = (ang_high and big_shift) and not (bimodal_bottom or bimodal_mid)
        return is_intersection_raw, is_curve90_raw

    def _strip_profile(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, self.LEFT_CROP:self.RIGHT_CROP]
        if roi.size == 0:
            return None
        colsum = roi.sum(axis=0).astype(np.float32)
        if colsum.max() > 0:
            colsum /= (255.0 * h)
        return colsum

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

    # ----------------- LOOP PRINCIPAL -----------------
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

                is_intersection_now, is_curve90_now = False, False
                try:
                    is_intersection_now, is_curve90_now = self._detect_intersection_core(bw, widths, cents, angle)
                except Exception:
                    pass

                # Debounce counters + logs por frame
                if is_intersection_now:
                    self._intersect_seen = min(self._intersect_seen + 1, 20)
                    log(f"Viu possível interseção (frame {self._intersect_seen}/{self.INTERSECT_DEBOUNCE})")
                else:
                    if self._intersect_seen > 0:
                        log(f"Interseção não sustentada, reset contagem ({self._intersect_seen} -> 0)")
                    self._intersect_seen = 0

                if is_curve90_now:
                    self._c90_seen = min(self._c90_seen + 1, 20)
                    log(f"Viu possível curva 90 (frame {self._c90_seen}/{self.C90_DEBOUNCE})")
                else:
                    if self._c90_seen > 0:
                        log(f"Curva90 não sustentada, reset contagem ({self._c90_seen} -> 0)")
                    self._c90_seen = 0

                green_centroids, green_dir = self.vision.detect_greens(frame)
                if green_dir:
                    self._green_last = green_dir
                    self._green_seen = min(self._green_seen + 1, 20)
                    log(f"Viu marcador verde: {green_dir} (frame {self._green_seen}/{self.GREEN_DEBOUNCE})")
                else:
                    if self._green_seen > 0:
                        log(f"Verde não sustentado, reset contagem ({self._green_seen} -> 0)")
                    self._green_seen = 0

                confirmed_intersection = self._intersect_seen >= self.INTERSECT_DEBOUNCE
                confirmed_curve90 = self._c90_seen >= self.C90_DEBOUNCE
                confirmed_green = self._green_last if self._green_seen >= self.GREEN_DEBOUNCE else None

                if confirmed_intersection and not confirmed_curve90:
                    log(f"CONFIRMED: interseção detectada (green={confirmed_green})")
                    if confirmed_green == "uturn":
                        # uturn: avança e gira 180° (duas vezes o tempo de giro 90)
                        self._forward_time(self.INTERSECT_FWD_TIME, reason="intersection_uturn_forward")
                        self._turn_in_place_time("left", self.TURN90_TURN_TIME * 2, reason="intersection_uturn_turn")
                        self._green_seen = 0; self._intersect_seen = 0; self._green_last = None
                        continue
                    elif confirmed_green in ("left", "right"):
                        log(f"CONFIRMED: interseção com indicação GREEN -> {confirmed_green}")
                        self._forward_time(self.TURN90_FWD_TIME, reason="intersection_green_forward")
                        self._turn_in_place_time(confirmed_green, self.TURN90_TURN_TIME, reason="intersection_green_turn")
                        self._green_seen = 0; self._intersect_seen = 0
                        continue
                    else:
                        log("CONFIRMED: interseção sem marcador verde -> segue reto um pouco")
                        self._forward_time(self.INTERSECT_FWD_TIME, reason="intersection_nogreen_forward")
                        self._intersect_seen = 0
                        continue

                # Se for curva 90 confirmada (sem marcar interseção) -> log e tratar
                if confirmed_curve90:
                    log("CONFIRMED: curva 90 detectada (vai executar giro padrão)")
                    # Aplicar giro padrão para curva (aqui escolhemos left se angle positivo, else right)
                    # Podemos inferir direção do ângulo: se angle>0 -> curva para a direita (depende da sua calibração)
                    # Para evitar confusão, prefira usar TURN_DIRECTION_INVERT no config
                    dir_guess = "left" if angle < 0 else "right"
                    log(f"Curva90: direção inferida por ângulo = {dir_guess} (angle={angle:.1f})")
                    self._turn_in_place_time(dir_guess, self.TURN90_TURN_TIME, reason="c90_infered_turn")
                    self._c90_seen = 0
                    continue

                # controle normal de seguimento de linha
                c0 = cents[0]
                if c0 is None:
                    self.hardware.stop()
                else:
                    offset = c0[0] - (self.WIDTH // 2)
                    error = float(offset) + self.MIX_ANGLE * float(angle)
                    self._drive(self.BASE_SPEED, error)

                # Publica frame e status
                SHARED_STATE["last_frame"] = frame
                SHARED_STATE["speeds"] = {"left": getattr(self.hardware, "last_left_speed", 0), "right": getattr(self.hardware, "last_right_speed", 0)}
                self._frames += 1
                now = time.time()
                if now - getattr(self, "_last_ts", now) >= 2.0:
                    fps = self._frames / (now - self._last_ts)
                    self._frames = 0
                    self._last_ts = now
                    SHARED_STATE["fps"] = round(fps, 1)
                    log(f"Status: FPS ~ {fps:.1f} | L/R: {SHARED_STATE['speeds']['left']}/{SHARED_STATE['speeds']['right']}")

                time.sleep(0.005)

        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass


# ----------------- Web/Socket -----------------
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
                    data = cmd.get("data", {}) or {}
                    if name == "start_robot":
                        robot.start()
                    elif name == "stop_robot":
                        robot.stop()
                    elif name == "set_view_mode":
                        robot.set_view_mode(data.get("mode", "preview"))
                    elif name == "calibrate_pixel":
                        robot.calibrate_pixel(int(data["x"]), int(data["y"]), data.get("color", "green"))
                    elif name == "save_config":
                        ok = robot.save_config(data or {})
                        log(f"save_config via socket -> {ok}")
                    else:
                        log(f"Comando desconhecido: {cmd}")
                except Exception as e:
                    log(f"Erro no comando via socket: {e}")
        else:
            log("web_stream sem app/socketio/register_robot; headless.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")


# ----------------- Botão físico -----------------
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
            def _toggle(channel, _pin=pin):
                try:
                    time.sleep(0.03)
                    if GPIO.input(_pin) == GPIO.LOW:
                        if robot.running:
                            log(f"Botão (BCM {_pin}): STOP"); robot.stop()
                        else:
                            log(f"Botão (BCM {_pin}): START"); robot.start()
                except Exception as e:
                    log(f"Erro no botão (BCM {_pin}): {e}")
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=_toggle, bouncetime=300)
            configured = True; chosen_pin = pin
            log(f"Botão físico pronto no BCM {pin} (pull-up, FALLING).")
            break
        except Exception as e:
            log(f"Falha ao configurar botão no BCM {pin}: {e}")
    if not configured:
        log("Nenhum botão configurado (BCM21/4). Siga pela UI/web.")
        return


# ----------------- main -----------------
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
            robot.stop()
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
            try: robot.stop()
            except Exception: pass
            try: GPIO.cleanup()
            except Exception: pass

if __name__ == "__main__":
    main()
