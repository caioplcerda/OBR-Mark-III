# main.py
# Estilo RCJ 2014 com defensivas: scanline/scancircle/derivada no loop,
# PID defaults, compat de set_motor_speed, UI web opcional e Picamera2/USB.

import os
import cv2
import json
import time
import threading
from datetime import datetime
from collections import deque

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
    "view_mode": "normal",
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
    Núcleo à la RCJ_2014.cpp:
      - ZERO scan (linha horizontal na base)
      - FIRST scan (círculo com look-ahead pelo ângulo estimado)
      - SECOND/THIRD scan (refinos de look-ahead)
      - erro composto = offset_x + k*ângulo
      - decisões por verde (left/right/uturn/straight) e chegada (vermelho)
    """
    # ==== Constantes de "pilotagem" no estilo RCJ ====
    WIDTH = 640
    HEIGHT = 480

    BASE_SPEED = 50          # velocidade base (ajuste conforme tua ponte H/motores)
    MIX_ANGLE = 0.6          # ganho do termo angular no erro composto
    MAX_ANGLE = 45.0         # clamp em graus

    ZERO_SCAN_Y = 420        # linha baixa para "zero scan" (imagem 480p, já girada 180°)
    ZERO_SCAN_RADIUS = 320   # metade da largura coberta no zero scan
    CIRCLE_RADIUS = 22       # raio pequeno do scancircle (ponto local)
    LOOK_WIDTH_DEG = 180     # janela angular usada no look-ahead

    # PID defaults (evita KeyError no HardwareControl.__init__)
    PID_DEFAULTS = {"kp": 0.8, "ki": 0.0, "kd": 0.12, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None
        self.state = "IDLE"

        # Componentes
        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=True)
        self.vision = Vision({}, log)  # (config, log_function)

        # IMPORTANTE: HardwareControl exige config com 'pid' completo no __init__
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        # Variáveis RCJ
        self.first_scanpoint = (self.WIDTH // 2, self.ZERO_SCAN_Y)
        self.first_angle_deg = 0.0
        self.line_points = deque(maxlen=7)

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
        """Persiste parâmetros vindos da UI e propaga para Vision/Hardware (defensivo)."""
        try:
            SHARED_STATE["config"].update(new_cfg or {})
            self._ensure_pid_defaults(SHARED_STATE["config"])

            with open(self.cfg_path, "w", encoding="utf-8") as f:
                json.dump(SHARED_STATE["config"], f, ensure_ascii=False, indent=2)

            # Visão
            if "vision" in SHARED_STATE["config"]:
                self.vision.update_config(SHARED_STATE["config"]["vision"])

            # Hardware (PID)
            try:
                if hasattr(self.hardware, "config") and isinstance(self.hardware.config, dict):
                    # mantém self.hardware.config sincronizado com SHARED_STATE["config"]
                    self.hardware.config.update(SHARED_STATE["config"])
                if hasattr(self.hardware, "update_pid_from_config"):
                    self.hardware.update_pid_from_config()
            except Exception as e:
                log(f"Aviso: falha ao propagar PID para HardwareControl: {e}")

            log("Config salva.")
            return True
        except Exception as e:
            log(f"Falha save_config: {e}")
            return False

    def _load_config_if_any(self):
        """Carrega config.json (se existir) e propaga para Vision/Hardware (defensivo)."""
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
        """
        Compat: se hardware_control.set_motor_speed aceitar (base, erro), usa direto.
        Se aceitar (left, right), converte.
        """
        try:
            # Tentativa 1: assinatura (base, erro)
            self.hardware.set_motor_speed(base_speed, error)
        except TypeError:
            # Tentativa 2: assinatura (left, right)
            left = int(max(-100, min(100, base_speed - error)))
            right = int(max(-100, min(100, base_speed + error)))
            self.hardware.set_motor_speed(left, right)

    # ====== Lógica principal estilo RCJ ======
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
                frame = self.camera.read()  # BGR (já rotacionado em Camera)
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
                    min_line_width=12,
                )
                derivative_export = deriv0

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
                    min_line_width=6,
                )
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
                    gray, self.line_points[0], self.CIRCLE_RADIUS, self.first_angle_deg, 180
                )
                p2, _ = rcj.find_line_from_scan(
                    scan2, angs2, "circle",
                    {"center_point": self.line_points[0], "radius": self.CIRCLE_RADIUS}, 6
                )
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
                    gray, self.line_points[0], self.CIRCLE_RADIUS, look2, 180
                )
                p3, _ = rcj.find_line_from_scan(
                    scan3, angs3, "circle",
                    {"center_point": self.line_points[0], "radius": self.CIRCLE_RADIUS}, 6
                )
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
                    self._drive(0, -120)  # gira
                    time.sleep(0.5)

                if red_detected:
                    status_msg = "Chegada (vermelho) detectada."
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass

                # ---------- (G) Publicação ----------
                self._publish(frame, status_msg, derivative_export,
                              extras={"green": green_dir, "greens": greens, "red": red_detected})

                time.sleep(0.005)  # folga CPU

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
        # Sem web: inicia imediatamente
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
