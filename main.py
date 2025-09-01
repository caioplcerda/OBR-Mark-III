# main.py
# Segue-linha robusto para LINHA GROSSA com:
# - multi-ROI (faixas horizontais) + regressão para ângulo
# - detecção de interseção por largura em múltiplas faixas (com debounce)
# - marcadores VERDES com ROI e filtro de forma (debounce)
# - ações: esquerda, direita, reto (sem marca), retorno (duplo verde)
# - Preview no stream (faixas, centróides, linha ajustada, look-ahead)
#
# Observação: o servidor web é fornecido por web_stream.py (importado aqui).
# Rode apenas:  python3 main.py   e acesse http://<ip-do-raspberry>:5000/

import os
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque

# ==== Estado Web (compartilhado) ====
WEB_AVAILABLE = False
SHARED_STATE = {
    "config": {},
    "last_frame": None,
    "speeds": {"left": 0, "right": 0},
    "view_mode": "preview",     # abre já em preview
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

# ==== Dependências do projeto ====
from hardware_control import HardwareControl
from vision import Vision  # HSV e detecção de verdes/vermelho

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
    """Picamera2 (preferido) ou USB/OpenCV. Rotaciona 180° (placa montada ao contrário)."""
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
    """Pipeline robusto para linha grossa com multi-ROI + interseção + verdes."""
    WIDTH = 640
    HEIGHT = 480

    # Controle
    BASE_SPEED = 55
    MIX_ANGLE = 0.7        # peso do termo angular no erro composto
    MAX_ANGLE = 50.0

    # Multi-ROI (faixas)
    N_STRIPS = 8           # nº de faixas horizontais
    STRIP_H = 22           # altura de cada faixa
    STRIP_BOTTOM = 440     # y do topo da faixa mais baixa

    # Limiares / morfologia
    BIN_BLUR = 3           # blur leve antes do binário
    ADAPT_BLOCK = 21       # adaptive threshold (ímpar)
    ADAPT_C = 7
    MORPH = 3              # abertura p/ quebrar “brilhos”

    # Interseção (largura de linha grossa)
    INTERSECTION_WIDTH_PX = 180    # largura mínima (px) em 640px para considerar interseção
    INTERSECT_DEBOUNCE = 2         # quadros consecutivos

    # Verdes (debounce)
    GREEN_DEBOUNCE = 2

    # PID defaults (evita KeyError no HardwareControl.__init__)
    PID_DEFAULTS = {"kp": 0.9, "ki": 0.0, "kd": 0.14, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        # Componentes
        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=True)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        # histórico de pontos p/ previsão suave
        self.history = deque(maxlen=5)  # últimos centróides da faixa base

        # buffers/estado
        self._intersect_seen = 0
        self._green_seen = 0
        self._green_last = None
        self.planned_direction = None  # "left"|"right"|"straight"|"uturn"|None
        self.turning_until = 0.0       # timestamp para ação de curva breve

        # Config persistente
        self.cfg_path = "config.json"
        self._load_config_if_any()

        # FPS
        self._last_ts = time.time()
        self._frames = 0

    # ===== ciclo de vida =====
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
        log("Parado.")

    def cleanup(self):
        self.running = False
        try:
            self.hardware.cleanup()
        except Exception:
            pass
        self.camera.release()
        log("Recursos liberados.")

    # ===== UI helpers =====
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
            # sincroniza PID se houver
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

    # ===== motor compat =====
    def _drive(self, base_speed: float, error: float):
        try:
            self.hardware.set_motor_speed(base_speed, error)  # (base, erro)
        except TypeError:
            left = int(max(-100, min(100, base_speed - error)))
            right = int(max(-100, min(100, base_speed + error)))
            self.hardware.set_motor_speed(left, right)        # (left, right)

    # ===== processamento da linha =====
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
        return bw  # 255 = linha (escuro), 0 = fundo

    def _strip_centroid(self, bw, y0, h):
        """centro de massa (x) da maior massa branca (linha) em uma faixa [y0:y0+h]."""
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, :]
        # soma por coluna
        colsum = roi.sum(axis=0)  # 255 por pixel ligado
        if colsum.max() < 255 * h * 0.05:  # pouquíssimos pixels -> vazio
            return None, 0
        x = np.arange(W, dtype=np.float32)
        cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
        cy = int((y0 + y1) / 2)
        # largura aproximada na linha média (px "ligados")
        width_est = int(np.count_nonzero(roi[(h//2)-1:(h//2)+1, :] > 0))
        return (int(cx), int(cy)), width_est

    def _fit_angle(self, pts):
        """ajusta x = alpha*y + beta (melhor p/ faixas horizontais); devolve ângulo (°) em relação ao eixo x."""
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        alpha, beta = np.linalg.lstsq(A, xs, rcond=None)[0]
        angle_deg = np.degrees(np.arctan2(alpha, 1.0))
        return float(np.clip(angle_deg, -self.MAX_ANGLE, self.MAX_ANGLE))

    def _draw_preview(self, frame, bw, cents, fit_angle, look_point, is_intersection, green_dir):
        out = frame.copy()

        # linha-base (vermelha) = topo da faixa mais baixa
        y_base = int(self.STRIP_BOTTOM)
        cv2.line(out, (0, y_base), (self.WIDTH-1, y_base), (0, 0, 255), 2)

        # faixas e centróides (verde)
        for i, c in enumerate(cents):
            y0 = int(self.STRIP_BOTTOM - i*self.STRIP_H)
            cv2.rectangle(out, (0, y0), (self.WIDTH-1, y0 + self.STRIP_H), (80, 80, 80), 1)
            if c is not None:
                cv2.circle(out, (int(c[0]), int(c[1])), 6, (0, 200, 0), -1, cv2.LINE_AA)

        # linha ajustada (azul)
        valids = [c for c in cents if c is not None]
        if len(valids) >= 2:
            pA = valids[0]
            pB = valids[-1]
            cv2.line(out, (int(pA[0]), int(pA[1])), (int(pB[0]), int(pB[1])), (255, 0, 0), 2, cv2.LINE_AA)

        # look-ahead (amarelo)
        if look_point is not None:
            cv2.circle(out, (int(look_point[0]), int(look_point[1])), 8, (0, 255, 255), 2, cv2.LINE_AA)

        # rótulos
        txt = f"angle={fit_angle:+.1f} strips={len(valids)}"
        if is_intersection:
            txt += "  INT"
        if green_dir:
            txt += f"  GREEN:{green_dir}"
        cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return out

    # ===== loop =====
    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()

                # --- binário da linha ---
                bw = self._binarize(frame)  # 255 = linha (escuro), 0 = fundo

                # --- centróides e larguras por faixas (de baixo pra cima) ---
                cents = []
                widths = []
                for i in range(self.N_STRIPS):
                    y0 = self.STRIP_BOTTOM - i*self.STRIP_H
                    c, w = self._strip_centroid(bw, y0, self.STRIP_H)
                    cents.append(c)
                    widths.append(w)

                # precisa pelo menos o da faixa mais baixa
                c0 = cents[0]
                if c0 is None:
                    self._publish(frame, "Linha perdida (faixa base vazia)")
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass
                    time.sleep(0.01)
                    continue

                # histórico
                self.history.append(c0[0])

                # regressão com pontos válidos
                valids = [p for p in cents if p is not None]
                angle = self._fit_angle(valids)

                # look-ahead: projeta um ponto adiante seguindo ângulo
                steps = 5
                dx = np.tan(np.radians(angle)) * steps * self.STRIP_H
                look = (c0[0] + dx, c0[1] - steps*self.STRIP_H)
                look = (int(np.clip(look[0], 0, self.WIDTH-1)),
                        int(np.clip(look[1], 0, self.HEIGHT-1)))

                # --- interseção por largura (linha grossa) com debounce ---
                wide_bottom = widths[0] >= self.INTERSECTION_WIDTH_PX
                wide_mid = (len(widths) > 2) and (widths[2] >= self.INTERSECTION_WIDTH_PX*0.85)
                is_intersection = wide_bottom and wide_mid
                if is_intersection:
                    self._intersect_seen = min(self._intersect_seen + 1, 10)
                else:
                    self._intersect_seen = 0
                confirmed_intersection = self._intersect_seen >= self.INTERSECT_DEBOUNCE

                # --- verdes (com ROI/filtros do Vision) + debounce ---
                green_centroids, green_dir = self.vision.detect_greens(frame)
                if green_dir:
                    self._green_last = green_dir
                    self._green_seen = min(self._green_seen + 1, 10)
                else:
                    self._green_seen = 0
                confirmed_green = self._green_last if self._green_seen >= self.GREEN_DEBOUNCE else None

                # --- decisões de interseção conforme regras OBR ---
                now = time.time()
                if confirmed_intersection:
                    if confirmed_green == "uturn":
                        # retorno: girar até reacoplar linha
                        log("UTURN: dois verdes detectados — iniciando retorno.")
                        t0 = time.time()
                        timeout = 3.0
                        while time.time() - t0 < timeout and self.running:
                            try:
                                self.hardware.set_motor_speed(0, 120)  # gira no lugar
                            except TypeError:
                                self.hardware.set_motor_speed(-60, 60)
                            # reacoplou? (muita massa de linha na base)
                            _, w0_re = self._strip_centroid(self._binarize(self.camera.read()), self.STRIP_BOTTOM, self.STRIP_H)
                            if w0_re > 40:
                                break
                        self.hardware.stop()
                        # depois do retorno, segue normal
                        self._green_seen = 0
                        self._intersect_seen = 0
                        self._green_last = None
                        self.planned_direction = None
                        continue

                    elif confirmed_green in ("left", "right"):
                        # curva guiada por verde: injeta viés por curto período
                        self.planned_direction = confirmed_green
                        self.turning_until = now + 0.7  # ~700 ms de viés
                        log(f"Curva {confirmed_green} marcada — aplicando viés por 0.7s.")
                        # reseta debounce
                        self._green_seen = 0
                        self._intersect_seen = 0

                    else:
                        # interseção sem marca: seguir reto (NADA a fazer)
                        self.planned_direction = "straight"
                        self.turning_until = now + 0.4  # manter reto por ~400ms
                        log("Interseção sem marca — seguindo reto.")

                # erro = offset_x + k*angulo (+ viés se houver plano de curva)
                offset = c0[0] - (self.WIDTH // 2)
                error = float(offset) + self.MIX_ANGLE * float(angle)

                # aplica viés temporário se houver curva planejada
                if self.planned_direction and now < self.turning_until:
                    bias = 120 if self.planned_direction == "left" else (-120 if self.planned_direction == "right" else 0)
                    error += bias
                elif self.planned_direction and now >= self.turning_until:
                    # encerra plano de curva
                    self.planned_direction = None

                # drive
                self._drive(self.BASE_SPEED, error)

                # preview opcional
                out = frame
                if SHARED_STATE.get("view_mode") == "preview":
                    out = self._draw_preview(frame, bw, cents, angle, look, confirmed_intersection, confirmed_green)

                # publica
                self._publish(out, "OK")

                time.sleep(0.003)

        except KeyboardInterrupt:
            log("Interrompido.")
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try:
                self.hardware.stop()
            except Exception:
                pass

    def _publish(self, frame, status_msg):
        left = getattr(self.hardware, "last_left_speed", 0)
        right = getattr(self.hardware, "last_right_speed", 0)
        SHARED_STATE["last_frame"] = frame
        SHARED_STATE["speeds"] = {"left": left, "right": right}
        SHARED_STATE["status"] = status_msg

        # FPS a cada ~2s
        self._frames += 1
        now = time.time()
        if now - getattr(self, "_last_ts", now) >= 2.0:
            fps = self._frames / (now - self._last_ts)
            self._frames = 0
            self._last_ts = now
            SHARED_STATE["fps"] = round(fps, 1)
            log(f"Status: {status_msg} | FPS ~ {fps:.1f} | L/R: {left}/{right}")


# ==== integração com web ====
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
                        robot.calibrate_pixel(int(data["x"]), int(data["y"]), data.get("color", "black"))
                    elif name == "save_config":
                        robot.save_config(data or {})
                    else:
                        log(f"Comando desconhecido: {cmd}")
                except Exception as e:
                    log(f"Erro no comando via socket: {e}")
        else:
            log("web_stream sem app/socketio/register_robot; seguindo headless.")
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
        log("Rodando headless (sem servidor web).")
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
