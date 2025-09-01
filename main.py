# main.py
# Estilo RCJ 2014 com defensivas e suporte a LINHA GROSSA:
# - scanline/scancircle/derivada no loop
# - fallbacks robustos (maior faixa escura e arco escuro)
# - modo PREVIEW (círculos/pontos/centroides/crosshair/linha-base)
# - PID defaults, compat de set_motor_speed, UI web opcional e Picamera2/USB.

import os
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque

# Se o vídeo aparecer com cores invertidas (verde vira roxo, etc), troque para True
FORCE_RGB_INPUT = False

# ==== Integração opcional com servidor web ====
WEB_AVAILABLE = False
SHARED_STATE = {
    "config": {},
    "last_frame": None,
    "mask": None,
    "contours": [],
    "path_history": [],
    "speeds": {"left": 0, "right": 0},
    "derivative_scan": None,
    "view_mode": "normal",  # normal | mask | contours | derivative | preview
    "status": "idle",
    "log": [],
}
try:
    import web_stream  # se existir no projeto
    if hasattr(web_stream, "SHARED_STATE"):
        SHARED_STATE = web_stream.SHARED_STATE
    WEB_AVAILABLE = True
except Exception:
    WEB_AVAILABLE = False

# ==== Dependências do projeto ====
from hardware_control import HardwareControl
from vision import Vision
import rcj2014_port as rcj

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
    """Abstrai Picamera2 (preferido) ou USB/OpenCV; rotaciona 180° por padrão para espelhar RCJ."""
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
    """
    Núcleo à la RCJ_2014.cpp com tolerância a LINHA GROSSA:
      - ZERO scan (linha horizontal na base)
      - FIRST/SECOND/THIRD (círculos com look-ahead)
      - fallback 1D/anel para linha grossa
      - erro composto = offset_x + k*ângulo
      - detecção verde/vermelho
      - modo PREVIEW
    """
    # ==== Constantes de "pilotagem" no estilo RCJ ====
    WIDTH = 640
    HEIGHT = 480

    BASE_SPEED = 55          # um pouco mais alto para vencer inércia
    MIX_ANGLE = 0.7          # ganho do termo angular no erro composto (linha grossa precisa mais direção)
    MAX_ANGLE = 50.0         # clamp em graus

    ZERO_SCAN_Y = 440        # mais perto da base para estabilizar 1º ponto
    ZERO_SCAN_RADIUS = 320   # metade da largura
    CIRCLE_RADIUS = 30       # raio maior ajuda linha grossa
    LOOK_WIDTH_DEG = 140     # janela mais estreita: reduz falsas escolhas

    # LARGURA MÍNIMA para RCJ (evita pegar apenas uma borda)
    ZERO_MIN_WIDTH = 28
    CIRCLE_MIN_WIDTH = 16

    # PID defaults
    PID_DEFAULTS = {"kp": 0.9, "ki": 0.0, "kd": 0.14, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None
        self.state = "IDLE"

        # Componentes
        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=True)
        self.vision = Vision({}, log)  # (config, log_function)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        # Variáveis RCJ
        self.first_scanpoint = (self.WIDTH // 2, self.ZERO_SCAN_Y)
        self.first_angle_deg = 0.0
        self.line_points = deque(maxlen=9)

        # Configuração persistente
        self.cfg_path = "config.json"
        self._load_config_if_any()

        # Medição de FPS
        self._last_ts = time.time()
        self._frames = 0

    # ===== Ciclo de vida =====
    def start(self):
        if self.running:
            log("Robô já está rodando.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.state = "FOLLOWING"
        SHARED_STATE["status"] = "running"
        log("Loop principal iniciado.")

    def stop(self):
        self.running = False
        self.state = "STOPPED"
        SHARED_STATE["status"] = "stopped"
        try:
            self.hardware.stop()
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
        log("Recursos liberados.")

    # ===== Utilidades / UI =====
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

    def _ensure_pid_defaults(self, cfg: dict):
        cfg.setdefault("pid", {})
        for k, v in self.PID_DEFAULTS.items():
            cfg["pid"].setdefault(k, v)

    def save_config(self, new_cfg: dict):
        try:
            SHARED_STATE["config"].update(new_cfg or {})
            self._ensure_pid_defaults(SHARED_STATE["config"])
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
                log(f"Aviso: falha ao propagar PID p/ HardwareControl: {e}")
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
                self._ensure_pid_defaults(cfg)
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

    # ====== Helpers de controle ======
    def _drive(self, base_speed: float, error: float):
        try:
            self.hardware.set_motor_speed(base_speed, error)  # (base, erro)
        except TypeError:
            left = int(max(-100, min(100, base_speed - error)))
            right = int(max(-100, min(100, base_speed + error)))
            self.hardware.set_motor_speed(left, right)        # (left, right)

    # ====== Fallbacks p/ LINHA GROSSA ======
    def _zero_scan_fallback(self, gray):
        """Escolhe a MAIOR faixa escura contínua na linha y=ZERO_SCAN_Y e usa o meio."""
        y = int(self.ZERO_SCAN_Y)
        row = gray[y, :].astype(np.float32)
        # suaviza para reduzir ruído
        k = 9
        row_s = np.convolve(row, np.ones(k)/k, mode="same")
        # limiar adaptativo (entre min e média)
        thr = 0.5*(row_s.min() + row_s.mean())
        dark = row_s < thr

        # encontra maiores segmentos True
        best_len, best_l, best_r = 0, None, None
        i = 0
        W = dark.shape[0]
        while i < W:
            if dark[i]:
                j = i
                while j < W and dark[j]:
                    j += 1
                seg_len = j - i
                if seg_len > best_len:
                    best_len, best_l, best_r = seg_len, i, j-1
                i = j
            else:
                i += 1

        if best_len >= self.ZERO_MIN_WIDTH:
            cx = (best_l + best_r) // 2
            return (int(cx), y)
        return None

    def _circle_scan_fallback(self, gray, center, look_deg, win_deg=140):
        """Amostra um anel e pega o ARCO mais escuro na janela em torno do look_deg."""
        cx, cy = int(center[0]), int(center[1])
        R = int(self.CIRCLE_RADIUS)
        # amostras por 360° (densidade suficientemente boa)
        N = 180
        ang = rcj.line_angle_from_points(self.line_points[0], self.line_points[1])
        # janela de look-ahead
        look2 = rcj.line_angle_from_points(self.line_points[0], self.line_points[1]) if len(self.line_points) > 1 else 0.0
        # normaliza ângulos relativos à janela
        def wrap(a):
            d = (a - look + np.pi) % (2*np.pi) - np.pi
            return d
        # escolhe apenas os ângulos dentro da janela
        half = np.deg2rad(win_deg/2)
        idx = np.where(np.abs(wrap(angs)) <= half)[0]
        if len(idx) < 8:
            return None

        sel_angs = angs[idx]
        vals = []
        for a in sel_angs:
            x = int(cx + R*np.cos(a))
            y = int(cy + R*np.sin(a))
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                vals.append(float(gray[y, x]))
            else:
                vals.append(255.0)
        vals = np.array(vals, dtype=np.float32)
        # suaviza
        k = max(5, len(vals)//25*2+1)  # ímpar
        ker = np.ones(k, dtype=np.float32)/k
        vals_s = np.convolve(vals, ker, mode="same")

        # limiar adaptativo e maior arco escuro
        thr = 0.5*(vals_s.min() + vals_s.mean())
        dark = vals_s < thr

        best_len, best_l, best_r = 0, None, None
        i = 0
        L = len(dark)
        while i < L:
            if dark[i]:
                j = i
                while j < L and dark[j]:
                    j += 1
                seg_len = j - i
                if seg_len > best_len:
                    best_len, best_l, best_r = seg_len, i, j-1
                i = j
            else:
                i += 1

        if best_len >= self.CIRCLE_MIN_WIDTH//2:
            mid = (best_l + best_r)//2
            a = sel_angs[mid]
            px = int(cx + R*np.cos(a))
            py = int(cy + R*np.sin(a))
            return (px, py)
        return None

    # ====== Desenho do PREVIEW (overlay) ======
    def _crosshair(self, img, pt, color=(0,165,255), size=10, thickness=2):
        if not pt:
            return
        x, y = int(pt[0]), int(pt[1])
        cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (x, y), size, color, thickness, cv2.LINE_AA)

    def _draw_preview(self, frame, p0, p1, p2, p3, greens, green_dir, angs1, angs2, angs3):
        overlay = frame.copy()
        # linha-base
        cv2.line(overlay, (0, self.ZERO_SCAN_Y), (self.WIDTH-1, self.ZERO_SCAN_Y), (0,0,255), 2)
        # círculo atual
        if len(self.line_points) > 0:
            cx, cy = self.line_points[0]
            cv2.circle(overlay, (int(cx), int(cy)), self.CIRCLE_RADIUS, (255,0,0), 1, cv2.LINE_AA)
        # anel pontilhado
        def _ring_points(angs):
            if angs is None or len(self.line_points) == 0:
                return
            cx, cy = self.line_points[0]
            step = max(1, len(angs)//12)
            for k in range(0, len(angs), step):
                x = int(cx + self.CIRCLE_RADIUS*np.cos(angs[k]))
                y = int(cy + self.CIRCLE_RADIUS*np.sin(angs[k]))
                cv2.circle(overlay, (x, y), 3, (255,255,0), -1, cv2.LINE_AA)
        _ring_points(angs1); _ring_points(angs2); _ring_points(angs3)
        # pontos dos scans
        for pt in [p0, p1, p2, p3]:
            if pt:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (0,0,255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 16, (255,0,0), 1, cv2.LINE_AA)
        # “trilho” verde (look-ahead)
        if len(self.line_points) > 0:
            cx, cy = self.line_points[0]
            look_rad = np.deg2rad(self.first_angle_deg)
            step = self.CIRCLE_RADIUS
            for t in range(1, 8):
                x = int(cx + t*step*np.cos(look_rad))
                y = int(cy + t*step*np.sin(look_rad))
                cv2.circle(overlay, (x, y), 6, (0,200,0), -1, cv2.LINE_AA)
        # centróides verdes + mira
        if greens:
            for (gx, gy) in greens:
                cv2.circle(overlay, (int(gx), int(gy)), 10, (0,255,0), 2, cv2.LINE_AA)
        self._crosshair(overlay, p0, (0,165,255), 12, 2)
        # status
        status = f"dir={green_dir or '-'} look={self.first_angle_deg:+.1f}"
        cv2.putText(overlay, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return overlay

    # ====== Lógica principal ======
    def _push_point(self, p):
        self.line_points.appendleft(p)
        try:
            SHARED_STATE["path_history"].append(p)
            if len(SHARED_STATE["path_history"]) > 400:
                SHARED_STATE["path_history"] = SHARED_STATE["path_history"][-400:]
        except Exception:
            pass

    def _detect_red(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.vision.LOWER_RED1, self.vision.UPPER_RED1)
        m2 = cv2.inRange(hsv, self.vision.LOWER_RED2, self.vision.UPPER_RED2)
        mask_red = cv2.add(m1, m2)
        return cv2.countNonZero(mask_red) > self.vision.GREEN_THRESHOLD_AREA

    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                status_msg = "OK"
                derivative_export = None

                # ---------- (A) ZERO SCAN ----------
                scan0, x0 = rcj.scanline(
                    gray,
                    (self.first_scanpoint[0], self.ZERO_SCAN_Y),
                    self.ZERO_SCAN_RADIUS,
                )
                p0, deriv0 = rcj.find_line_from_scan(
                    scan0,
                    x0,
                    "line",
                    {"center_point": (self.first_scanpoint[0], self.ZERO_SCAN_Y),
                     "radius": self.ZERO_SCAN_RADIUS},
                    min_line_width=self.ZERO_MIN_WIDTH,
                )
                derivative_export = deriv0

                # Fallback para linha grossa
                if not p0:
                    p0 = self._zero_scan_fallback(gray)

                if not p0:
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass
                    status_msg = "Linha perdida (zero scan)."
                    self._publish(frame, status_msg, derivative_export)
                    time.sleep(0.01)
                    continue

                self.first_scanpoint = p0
                self._push_point(p0)

                # ---------- (B) FIRST SCAN ----------
                scan1, angs1 = rcj.scancircle(
                    gray,
                    self.line_points[0],
                    self.CIRCLE_RADIUS,
                    self.first_angle_deg,
                    self.LOOK_WIDTH_DEG,
                )
                p1, _ = rcj.find_line_from_scan(
                    scan1, angs1, "circle",
                    {"center_point": self.line_points[0], "radius": self.CIRCLE_RADIUS},
                    min_line_width=self.CIRCLE_MIN_WIDTH,
                )
                if not p1:
                    p1 = self._circle_scan_fallback(gray, self.line_points[0], self.first_angle_deg, self.LOOK_WIDTH_DEG)

                if not p1:
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass
                    status_msg = "Linha perdida (first scan)."
                    self._publish(frame, status_msg, derivative_export)
                    time.sleep(0.01)
                    continue
                self._push_point(p1)

                # Atualiza ângulo inicial com clamp
                if len(self.line_points) > 1:
                    ang = rcj.line_angle_from_points(self.line_points[1], self.line_points[0])
                    self.first_angle_deg = max(-self.MAX_ANGLE, min(self.MAX_ANGLE, ang))

                # ---------- (C) SECOND SCAN ----------
                scan2, angs2 = rcj.scancircle(
                    gray, self.line_points[0], self.CIRCLE_RADIUS, self.first_angle_deg, 120
                )
                p2, _ = rcj.find_line_from_scan(
                    scan2, angs2, "circle",
                    {"center_point": self.line_points[0], "radius": self.CIRCLE_RADIUS},
                    min_line_width=self.CIRCLE_MIN_WIDTH,
                )
                if not p2:
                    p2 = self._circle_scan_fallback(gray, self.line_points[0], self.first_angle_deg, 120)

                if not p2:
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass
                    status_msg = "Linha perdida (second scan)."
                    self._publish(frame, status_msg, derivative_export)
                    time.sleep(0.01)
                    continue
                self._push_point(p2)

                # ---------- (D) THIRD SCAN ----------
                look2 = rcj.line_angle_from_points(self.line_points[1], self.line_points[0]) if len(self.line_points) > 1 else 0.0
                scan3, angs3 = rcj.scancircle(
                    gray, self.line_points[0], self.CIRCLE_RADIUS, look2, 120
                )
                p3, _ = rcj.find_line_from_scan(
                    scan3, angs3, "circle",
                    {"center_point": self.line_points[0], "radius": self.CIRCLE_RADIUS},
                    min_line_width=self.CIRCLE_MIN_WIDTH,
                )
                if not p3:
                    p3 = self._circle_scan_fallback(gray, self.line_points[0], look2, 120)

                if not p3:
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass
                    status_msg = "Linha perdida (third scan)."
                    self._publish(frame, status_msg, derivative_export)
                    time.sleep(0.01)
                    continue
                self._push_point(p3)

                # ---------- (E) Controle ----------
                P_err = self.first_scanpoint[0] - (self.WIDTH // 2)
                I_err = rcj.line_angle_from_points(self.line_points[1], self.line_points[0]) if len(self.line_points) > 1 else 0.0
                err_comp = float(P_err) + self.MIX_ANGLE * float(I_err)
                self._drive(self.BASE_SPEED, err_comp)

                # ---------- (F) Eventos ----------
                greens, green_dir, _mask_green = rcj.track_green_centroids(
                    frame,
                    {"lower": self.vision.LOWER_GREEN, "upper": self.vision.UPPER_GREEN},
                    area_min=self.vision.GREEN_THRESHOLD_AREA,
                )
                red_detected = self._detect_red(frame)

                if green_dir == "uturn":
                    log("Marcadores verdes dos dois lados: U-turn.")
                    self._drive(0, -120)
                    time.sleep(0.5)

                if red_detected:
                    status_msg = "Chegada (vermelho) detectada."
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass

                # ---------- (G) PREVIEW / PUBLICAÇÃO ----------
                frame_out = frame
                if SHARED_STATE.get("view_mode") == "preview":
                    frame_out = self._draw_preview(
                        frame, p0, p1, p2, p3, greens, green_dir, angs1, angs2, angs3
                    )

                self._publish(frame_out, status_msg, derivative_export,
                              extras={"green": green_dir, "greens": greens, "red": red_detected})

                time.sleep(0.005)

        except KeyboardInterrupt:
            log("Interrompido.")
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass

    def _publish(self, frame, status_msg, derivative_data, extras=None):
        left = getattr(self.hardware, "last_left_speed", 0)
        right = getattr(self.hardware, "last_right_speed", 0)
        SHARED_STATE["last_frame"] = frame
        SHARED_STATE["speeds"] = {"left": left, "right": right}
        SHARED_STATE["status"] = status_msg
        SHARED_STATE["derivative_scan"] = (
            derivative_data.tolist() if derivative_data is not None else None
        )
        # Atualiza FPS a cada ~2s
        self._frames += 1
        now = time.time()
        if now - getattr(self, "_last_ts", now) >= 2.0:
            fps = self._frames / (now - self._last_ts)
            self._frames = 0
            self._last_ts = now
            SHARED_STATE["fps"] = round(fps, 1)
            log(f"Status: {status_msg} | FPS ~ {fps:.1f} | L/R: {left}/{right}")
        if extras:
            for k, v in extras.items():
                SHARED_STATE[k] = v


# ==== Integração com web_stream (se existir) ====
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
                        robot.set_view_mode(data.get("mode", "normal"))
                    elif name == "calibrate_pixel":
                        robot.calibrate_pixel(int(data["x"]), int(data["y"]), data.get("color", "black"))
                    elif name == "save_config":
                        robot.save_config(data or {})
                    else:
                        log(f"Comando desconhecido: {cmd}")
                except Exception as e:
                    log(f"Erro no comando via socket: {e}")
        else:
            log("web_stream sem app/socketio/register_robot; seguindo headless+start manual.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")


def main():
    robot = Robot()
    _wire_web(robot)

    if WEB_AVAILABLE and hasattr(web_stream, "app"):
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5000"))
        log(f"Servidor web em http://{host}:{port}")
        try:
            if hasattr(web_stream, "socketio"):
                web_stream.socketio.run(web_stream.app, host=host, port=port, allow_unsafe_werkzeug=True)
            else:
                web_stream.app.run(host=host, port=port)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()
    else:
        log("Rodando em modo headless (sem servidor web).")
        try:
            robot.start()
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()


if __name__ == "__main__":
    main()
