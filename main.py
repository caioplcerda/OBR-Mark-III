# main.py
# Orquestrador do robô seguidor de linha com integração ao RCJ 2014 (scanline/scancircle/derivada),
# interface web (se disponível) e controle de hardware (motores/encoders/PID).

import os
import sys
import cv2
import json
import time
import threading
from datetime import datetime

# ===== Tentativa de integrar com o servidor web, se presente =====
WEB_AVAILABLE = False
SHARED_STATE = {
    "config": {},
    "last_frame": None,              # frame BGR mais recente
    "mask": None,                    # opcional: publicar alguma máscara
    "contours": [],                  # opcional
    "path_history": [],              # lista de pontos (x,y)
    "speeds": {"left": 0, "right": 0},
    "derivative_scan": None,         # vetor 1D para o modo "Derivatives"
    "view_mode": "normal",           # normal | mask | contours | derivative
    "status": "idle",
    "log": [],
}

try:
    import web_stream  # precisa existir no mesmo diretório
    if hasattr(web_stream, "SHARED_STATE"):
        SHARED_STATE = web_stream.SHARED_STATE  # usa o compartilhado do servidor
    WEB_AVAILABLE = True
except Exception:
    WEB_AVAILABLE = False


# ===== Dependências internas do projeto =====
from hardware_control import HardwareControl
from vision import Vision
from line_follower import LineFollower

# ===== Picamera2 (opcional) =====
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    from libcamera import Transform
    PICAMERA_AVAILABLE = True
except Exception:
    PICAMERA_AVAILABLE = False


def log(msg: str):
    """Log simples que também alimenta o painel web (se houver)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        SHARED_STATE["log"].append(line)
        # limita tamanho do log no painel
        if len(SHARED_STATE["log"]) > 300:
            SHARED_STATE["log"] = SHARED_STATE["log"][-300:]
    except Exception:
        pass


class Camera:
    """Abstração de câmera: usa Picamera2 se disponível; senão, fallback para OpenCV."""
    def __init__(self, width=640, height=480, rotate_180=True):
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.picam = None
        self.cap = None

        if PICAMERA_AVAILABLE:
            try:
                self.picam = Picamera2()
                config = self.picam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    transform=Transform(hflip=0, vflip=0)
                )
                self.picam.configure(config)
                self.picam.start()
                log("Picamera2 iniciada.")
            except Exception as e:
                log(f"Falha ao iniciar Picamera2: {e}. Usando OpenCV/USB.")
                self.picam = None

        if self.picam is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError("Não foi possível abrir nenhuma câmera.")

    def read(self):
        if self.picam is not None:
            # Picamera2 retorna RGB; converte para BGR
            frame = self.picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ok, frame_bgr = self.cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler frame da câmera USB.")
        if self.rotate_180:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        return frame_bgr

    def release(self):
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


class Robot:
    """Classe principal que integra câmera, visão, seguidor e hardware."""
    def __init__(self):
        self.running = False
        self.thread = None
        self.view_mode = "normal"  # controlado pela UI
        self.state = "IDLE"

        # Componentes
        self.camera = Camera(width=640, height=480, rotate_180=True)
        self.vision = Vision(log)
        self.hardware = HardwareControl(log)
        self.follower = LineFollower(self.hardware, self.vision, SHARED_STATE, log)

        # Config (carrega se existir)
        self.cfg_path = "config.json"
        self._load_config_if_any()

        # FPS medição
        self._last_ts = time.time()
        self._frames = 0

    # ------------- ciclo de vida -------------
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
        self.hardware.stop()
        log("Loop principal parado.")

    def cleanup(self):
        self.running = False
        try:
            self.hardware.cleanup()
        except Exception:
            pass
        try:
            self.camera.release()
        except Exception:
            pass
        log("Recursos liberados.")

    # ------------- utilidades -------------
    def set_view_mode(self, mode: str):
        self.view_mode = mode
        SHARED_STATE["view_mode"] = mode
        log(f"View mode -> {mode}")

    def calibrate_pixel(self, x, y, color: str):
        """Recebe clique na imagem (coordenadas 640x480) e calibra HSV."""
        try:
            # Usa último frame disponível para amostrar o pixel
            frame = SHARED_STATE.get("last_frame", None)
            if frame is None:
                log("Sem frame para calibrar.")
                return False
            # Garantir limites
            x = max(0, min(frame.shape[1] - 1, int(x)))
            y = max(0, min(frame.shape[0] - 1, int(y)))
            ok = self.vision.calibrate_by_click(frame, x, y, color)
            if ok:
                log(f"Calibração por clique ({color}) aplicada em ({x},{y}).")
            else:
                log("Calibração por clique falhou (retorno falso).")
            return ok
        except Exception as e:
            log(f"Erro em calibrate_pixel: {e}")
            return False

    def save_config(self, new_cfg: dict):
        """Persiste parâmetros vindos da UI (ex.: limites HSV, thresholds da detecção, PID etc.)."""
        try:
            SHARED_STATE["config"].update(new_cfg or {})
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                json.dump(SHARED_STATE["config"], f, ensure_ascii=False, indent=2)
            # Envia partes relevantes para os módulos
            self.vision.update_config(SHARED_STATE["config"].get("vision", {}))
            self.hardware.update_pid_from_config(SHARED_STATE["config"].get("pid", {}))
            log("Configuração salva em config.json.")
            return True
        except Exception as e:
            log(f"Falha ao salvar config: {e}")
            return False

    def _load_config_if_any(self):
        if os.path.exists(self.cfg_path):
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                SHARED_STATE["config"] = cfg
                self.vision.update_config(cfg.get("vision", {}))
                self.hardware.update_pid_from_config(cfg.get("pid", {}))
                log("Configuração carregada de config.json.")
            except Exception as e:
                log(f"Falha ao carregar config.json: {e}")

    # ------------- loop principal -------------
    def _loop(self):
        try:
            while self.running:
                frame = self.camera.read()  # BGR (já rotacionado se preciso)

                # Executa uma etapa do seguidor
                status_msg, status = self.follower.step(frame)

                # Atualiza velocidades para UI, se o HardwareControl expõe
                left = getattr(self.hardware, "last_left_speed", 0)
                right = getattr(self.hardware, "last_right_speed", 0)
                SHARED_STATE["speeds"] = {"left": left, "right": right}

                # Publica frame + telemetria para o painel
                SHARED_STATE["last_frame"] = frame
                SHARED_STATE["derivative_scan"] = status.get("derivative_scan", None)
                # Path/history opcional: se LineFollower preencher (use SS['path_history'] do lado dele)
                # Máscaras e contornos podem ser publicados por Vision ou pelo LineFollower conforme desejado

                # Status textual
                SHARED_STATE["status"] = status_msg

                # FPS log leve
                self._frames += 1
                now = time.time()
                if now - self._last_ts >= 2.0:
                    fps = self._frames / (now - self._last_ts)
                    self._frames = 0
                    self._last_ts = now
                    SHARED_STATE["fps"] = round(fps, 1)
                    # evita spam excessivo
                    log(f"Status: {status_msg} | FPS ~ {fps:.1f} | Speeds L/R: {left}/{right}")

                # Pequena folga de CPU
                time.sleep(0.005)

        except KeyboardInterrupt:
            log("Interrompido por teclado.")
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            self.hardware.stop()


# ===== Integração com o servidor web (se disponível) =====
def _wire_web(robot: Robot):
    """
    Faz o 'handshake' com web_stream, registrando callbacks de comandos.
    Este código tenta usar funções/convencões comuns; ajuste se seu web_stream expõe nomes diferentes.
    """
    if not WEB_AVAILABLE:
        return

    # Se o web_stream tiver um registrador de robô, use-o.
    # Caso contrário, registramos handlers diretamente no objeto socketio se existir.
    try:
        if hasattr(web_stream, "register_robot"):
            web_stream.register_robot(robot)
            log("Robô registrado no servidor web.")
        elif hasattr(web_stream, "socketio"):
            # Fallback genérico: cria listeners básicos de comando
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

            log("Listeners de comando básicos conectados ao Socket.IO.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")


def main():
    robot = Robot()
    _wire_web(robot)

    if WEB_AVAILABLE:
        # Sobe o servidor web na *main thread*; o loop do robô roda em thread separada quando 'start' for chamado.
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5000"))
        log(f"Servidor web online em http://{host}:{port}")
        try:
            # Preferir socketio.run se existir
            if hasattr(web_stream, "socketio") and hasattr(web_stream, "app"):
                web_stream.socketio.run(web_stream.app, host=host, port=port, allow_unsafe_werkzeug=True)
            elif hasattr(web_stream, "app"):
                web_stream.app.run(host=host, port=port)
            else:
                log("web_stream não expõe 'app' nem 'socketio'; rodando headless.")
                robot.start()
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()
    else:
        # Sem web: roda direto o loop
        log("Servidor web não encontrado; rodando em modo headless.")
        try:
            robot.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            robot.cleanup()


if __name__ == "__main__":
    main()
