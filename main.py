# main.py corrigido (giro incremental substituído para evitar overshoot 180°)
# Substitui _turn_until_line por uma versão incremental em bursts.

import os
import cv2
import time
import threading
import numpy as np
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

    INTERSECT_DEBOUNCE = 6   # aumentado
    INTERSECT_AHEAD_DEBOUNCE = 4
    C90_DEBOUNCE = 6          # aumentado
    GREEN_DEBOUNCE = 2

    # tempos base (servirão como timeouts máximos no novo método)
    INTERSECT_FWD_TIME = 0.8
    TURN90_FWD_TIME = 0.4
    TURN90_TURN_TIME = 1.2  # tempo máximo aceitável para 90 (usado como timeout)

    # bias de giro (reduzido para evitar giros bruscos)
    TURN_BIAS = 100  # valor +/- para giro in-place

    PID_DEFAULTS = {"kp": 0.6, "ki": 0.0, "kd": 0.1, "sample_time": 0.02}

    def __init__(self):
        self.running = False
        self.thread = None

        self.camera = Camera(self.WIDTH, self.HEIGHT, rotate_180=False)
        self.vision = Vision({}, log)
        self.hardware = HardwareControl({"pid": dict(self.PID_DEFAULTS)})

        self.history = deque(maxlen=5)
        self._intersect_seen = 0
        self._c90_seen = 0
        self._green_seen = 0
        self._green_last = None
        self.planned_direction = None

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
        try: self.hardware.stop()
        except Exception: pass
        log("Parado.")

    def _drive(self, base_speed: float, error: float):
        # mesma compatibilidade que já tinha
        try:
            if hasattr(self.hardware, "set_motor_speed"):
                self.hardware.set_motor_speed(base_speed, error)
            elif hasattr(self.hardware, "drive"):
                left = int(base_speed - error)
                right = int(base_speed + error)
                self.hardware.drive(left, right)
            elif hasattr(self.hardware, "set_motors"):
                self.hardware.set_motors(base_speed, error)
            else:
                if hasattr(self.hardware, "write_pwm"):
                    try:
                        left = int(base_speed - error)
                        right = int(base_speed + error)
                        self.hardware.write_pwm(left_cmd=float(left), right_cmd=float(right))
                    except Exception:
                        pass
        except Exception as e:
            log(f"Falha ao enviar comando _drive: {e}")

    def _forward_time(self, duration_s: float):
        try:
            if hasattr(self.hardware, "set_motor_speed"):
                self.hardware.set_motor_speed(self.BASE_SPEED, 0)
            elif hasattr(self.hardware, "drive"):
                self.hardware.drive(self.BASE_SPEED, self.BASE_SPEED)
            else:
                try:
                    self.hardware.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                except Exception:
                    pass
            start_time = time.time()
            while self.running:
                if time.time() - start_time >= duration_s:
                    break
                time.sleep(0.01)
        finally:
            try: self.hardware.stop()
            except Exception: pass

    def _turn_in_place_time(self, direction: str, duration_s: float):
        bias = self.TURN_BIAS if direction == "left" else -self.TURN_BIAS
        start = time.time()
        try:
            while self.running and (time.time() - start) < duration_s:
                try:
                    if hasattr(self.hardware, "set_motor_speed"):
                        self.hardware.set_motor_speed(0, bias)
                    elif hasattr(self.hardware, "drive"):
                        l = int(-bias/2); r = int(bias/2)
                        self.hardware.drive(l, r)
                    elif hasattr(self.hardware, "set_motors"):
                        self.hardware.set_motors(0, bias)
                    elif hasattr(self.hardware, "write_pwm"):
                        self.hardware.write_pwm(left_cmd=float(-bias), right_cmd=float(bias))
                except Exception:
                    pass
                time.sleep(0.02)
        finally:
            try: self.hardware.stop()
            except Exception: pass

    def _turn_until_line(self, direction: str, timeout_s: float):
        """
        Gira em pequenos bursts até detectar a linha nos strips inferiores.
        - direction: "left" ou "right"
        - timeout_s: timeout máximo total
        Retorna True se encontrou a linha, False se timeout.
        """
        if direction not in ("left", "right"):
            direction = "left"
        sign = 1 if direction == "left" else -1

        base_bias = max(40, min(self.TURN_BIAS, 160))  # limites defensivos
        step_time = 0.12            # tempo de cada burst (ajuste fino)
        check_dt = 0.03             # intervalo de amostragem dentro do burst
        max_steps = max(1, int(timeout_s / step_time))
        found = False
        start_time = time.time()

        log(f"Iniciando giro incremental {direction} timeout={timeout_s:.2f}s steps={max_steps}")

        try:
            for step in range(max_steps):
                # rampa suave: começa mais suave e aumenta se não encontrar
                ramp = 0.5 + 0.5 * (step / max_steps)   # entre 0.5 e 1.0
                bias = int(sign * base_bias * ramp)

                # aplica o comando de giro por pequenos intervalos e verifica frame a cada check_dt
                burst_start = time.time()
                while (time.time() - burst_start) < step_time and (time.time() - start_time) < timeout_s and self.running:
                    try:
                        if hasattr(self.hardware, "set_motor_speed"):
                            self.hardware.set_motor_speed(0, bias)
                        elif hasattr(self.hardware, "drive"):
                            l = int(-bias/2); r = int(bias/2)
                            self.hardware.drive(l, r)
                        elif hasattr(self.hardware, "set_motors"):
                            self.hardware.set_motors(0, bias)
                        elif hasattr(self.hardware, "write_pwm"):
                            try:
                                self.hardware.write_pwm(left_cmd=float(-bias), right_cmd=float(bias))
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # pega frame e checa reaparição da linha nos 2 strips inferiores
                    try:
                        frame = self.camera.read()
                        bw = self._binarize(frame)
                        for k in range(2):   # verificar 2 strips inferiores
                            y0 = self.STRIP_BOTTOM - k*self.STRIP_H
                            c, w = self._strip_centroid(bw, y0, self.STRIP_H)
                            if c is not None and w > 0:
                                found = True
                                log(f"Linha encontrada durante giro (step {step}, burst): strip={k} c={c} w={w}")
                                break
                        if found:
                            break
                    except Exception:
                        # ignorar falha de frame e continuar
                        pass

                    time.sleep(check_dt)

                if found or (time.time() - start_time) >= timeout_s or not self.running:
                    break

            # se não encontrou, tenta um pequeno movimento compensatório no sentido oposto (anti-stuck)
            if not found and self.running:
                log("Não encontrou linha: tentativa rápida oposta (anti-stuck)")
                # curto burst oposto
                opp_sign = -sign
                opp_bias = int(opp_sign * base_bias * 0.6)
                try:
                    opp_start = time.time()
                    while time.time() - opp_start < 0.4 and self.running:
                        try:
                            if hasattr(self.hardware, "set_motor_speed"):
                                self.hardware.set_motor_speed(0, opp_bias)
                            elif hasattr(self.hardware, "drive"):
                                l = int(-opp_bias/2); r = int(opp_bias/2)
                                self.hardware.drive(l, r)
                            elif hasattr(self.hardware, "set_motors"):
                                self.hardware.set_motors(0, opp_bias)
                        except Exception:
                            pass
                        try:
                            frame = self.camera.read()
                            bw = self._binarize(frame)
                            for k in range(2):
                                y0 = self.STRIP_BOTTOM - k*self.STRIP_H
                                c, w = self._strip_centroid(bw, y0, self.STRIP_H)
                                if c is not None and w > 0:
                                    found = True
                                    log(f"Linha encontrada durante anti-stuck oposto: strip={k} c={c} w={w}")
                                    break
                            if found:
                                break
                        except Exception:
                            pass
                        time.sleep(check_dt)
                finally:
                    try: self.hardware.stop()
                    except Exception: pass

        finally:
            try: self.hardware.stop()
            except Exception: pass

        if found:
            # dá um pequeno avanço para estabilizar sobre a linha (reduz overshoot no retorno ao loop)
            try:
                log("Avançando um pouco para estabilizar após encontrar a linha")
                if hasattr(self.hardware, "set_motor_speed"):
                    self.hardware.set_motor_speed(int(self.BASE_SPEED*0.6), 0)
                elif hasattr(self.hardware, "drive"):
                    self.hardware.drive(int(self.BASE_SPEED*0.6), int(self.BASE_SPEED*0.6))
                time.sleep(0.12)
            except Exception:
                pass
            try: self.hardware.stop()
            except Exception: pass
            return True

        log("Timeout sem encontrar linha no giro incremental.")
        return False

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

                is_intersection, is_curve90, curve_dir = self._detect_intersection(bw, widths, cents, angle)

                if is_curve90:
                    self._c90_seen += 1
                else:
                    self._c90_seen = 0
                confirmed_curve90 = self._c90_seen >= self.C90_DEBOUNCE

                if is_intersection:
                    self._intersect_seen += 1
                else:
                    self._intersect_seen = 0
                confirmed_intersection = self._intersect_seen >= self.INTERSECT_DEBOUNCE

                # DETECÇÃO DE VERDE (usa Vision.detect_greens se existir; fallback interno caso contrário)
                green_centroids, green_dir = self._detect_greens(frame)
                if green_dir:
                    self._green_last = green_dir
                    self._green_seen += 1
                else:
                    self._green_seen = 0
                confirmed_green = self._green_last if self._green_seen >= self.GREEN_DEBOUNCE else None

                # Prioridade: curvas 90 confirmadas
                if confirmed_curve90:
                    dir_to_turn = curve_dir or confirmed_green or self._infer_curve_direction(valids, angle)
                    log(f"Curva 90° confirmada -> direção: {dir_to_turn}")
                    # anda um pouco e gira até encontrar linha (timeout = TURN90_TURN_TIME)
                    self._forward_time(self.TURN90_FWD_TIME)
                    found = self._turn_until_line(dir_to_turn, timeout_s=self.TURN90_TURN_TIME)
                    if not found:
                        # fallback: girar por tempo reduzido (evita 180°)
                        fallback_time = min(self.TURN90_TURN_TIME, 0.9)
                        log(f"Fallback: girar por tempo fixo {fallback_time:.2f}s")
                        self._turn_in_place_time(direction=dir_to_turn, duration_s=fallback_time)
                    self._c90_seen = 0
                    self._green_seen = 0
                    self._intersect_seen = 0
                    self._green_last = None
                    continue

                # Interseções
                if confirmed_intersection and not confirmed_curve90:
                    if confirmed_green == "uturn":
                        # Additional safety: only do U-turn if we've seen the signal for multiple frames
                        if self._green_seen >= 4:  # Require more confirmation for U-turns
                            log("Interseção com sinal verde (UTURN) detectada -> executando U-turn")
                            self._forward_time(self.INTERSECT_FWD_TIME)
                            # U-turn: tenta girar até achar linha duas vezes, timeout maior
                            found = self._turn_until_line("left", timeout_s=self.TURN90_TURN_TIME * 3)
                            if not found:
                                # se não achou, fallback para tempo (maior)
                                self._turn_in_place_time("left", duration_s=self.TURN90_TURN_TIME * 2)
                            self._green_seen = 0; self._intersect_seen = 0; self._green_last = None
                            continue
                        else:
                            log("Sinal verde U-turn detectado mas insuficiente confirmação - seguindo em frente")
                            self._forward_time(self.INTERSECT_FWD_TIME)
                            self._intersect_seen = 0
                            continue
                    elif confirmed_green in ("left", "right"):
                        log(f"Interseção com sinal verde ({confirmed_green}) -> executando curva")
                        self._forward_time(self.TURN90_FWD_TIME)
                        found = self._turn_until_line(confirmed_green, timeout_s=self.TURN90_TURN_TIME)
                        if not found:
                            # fallback curto
                            self._turn_in_place_time(confirmed_green, duration_s=min(self.TURN90_TURN_TIME, 0.9))
                        self._green_seen = 0; self._intersect_seen = 0
                        continue
                    else:
                        log("Interseção (sem verde) -> seguir em frente por pequeno tempo")
                        self._forward_time(self.INTERSECT_FWD_TIME)
                        self._intersect_seen = 0
                        continue

                # Seguimento normal
                if valids:
                    offset = valids[0][0] - (self.WIDTH // 2)
                    error = float(offset) + self.MIX_ANGLE * float(angle)
                    self._drive(self.BASE_SPEED, error)
                else:
                    log("Linha perdida: parando e esperando (ou implementar busca mais avançada)")
                    try:
                        self.hardware.stop()
                    except Exception:
                        pass

                # atualiza estado para UI
                left_spd = getattr(self.hardware, "last_left_speed", 0)
                right_spd = getattr(self.hardware, "last_right_speed", 0)
                SHARED_STATE["last_frame"] = frame
                SHARED_STATE["speeds"] = {"left": left_spd, "right": right_spd}
                time.sleep(0.01)
        except Exception as e:
            log(f"Erro no loop principal: {e}")
        finally:
            try: self.hardware.stop()
            except Exception: pass

    # ==== visão utilitários (mantidos) ====
    def _binarize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
        return bw

    def _strip_centroid(self, bw, y0, h):
        H, W = bw.shape[:2]
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y0 + h)))
        roi = bw[y0:y1, :]
        colsum = roi.sum(axis=0)
        threshold = 255 * h * 0.03
        if colsum.max() < threshold:
            return None, 0
        x = np.arange(0, W, dtype=np.float32)
        cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
        cy = int((y0 + y1) / 2)
        return (int(cx), int(cy)), int(np.count_nonzero(colsum))

    def _fit_angle(self, pts):
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        alpha, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        angle_deg = np.degrees(np.arctan2(alpha, 1.0))
        return float(np.clip(angle_deg, -self.MAX_ANGLE, self.MAX_ANGLE))

    def _detect_intersection(self, bw, widths, cents, angle_deg):
        wide = sum(widths[:2]) > 300
        ang_high = abs(angle_deg) > 35
        valids = [p for p in cents if p is not None]
        curve_dir = None
        if valids:
            xs = [p[0] for p in valids]
            avg_x = sum(xs) / len(xs)
            if avg_x < self.WIDTH * 0.25:
                curve_dir = "left"
            elif avg_x > self.WIDTH * 0.75:
                curve_dir = "right"
        total_width = sum(widths)
        disappearance = total_width < 50 and len([w for w in widths if w > 0]) < 2
        
        # Only detect curve90 if there's clear evidence of a curve, not just high angle
        # At intersections, high angle alone shouldn't trigger curve90
        is_curve90 = (curve_dir is not None and disappearance) or (disappearance and abs(angle_deg) > 20)
        
        if ang_high and curve_dir is None:
            curve_dir = "left" if angle_deg < 0 else "right"
        return wide, is_curve90, curve_dir

    def _detect_greens(self, frame):
        try:
            if hasattr(self.vision, "detect_greens"):
                out = self.vision.detect_greens(frame)
                if isinstance(out, tuple) and len(out) >= 2:
                    return out[0], out[1]
                elif isinstance(out, str) or out is None:
                    return [], out
        except Exception as e:
            log(f"vision.detect_greens falhou: {e}")

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 60, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            if not contours:
                return [], None
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            c = contours[max_idx]
            area = areas[max_idx]
            M = cv2.moments(c)
            if M.get("m00", 0) <= 0:
                return [], None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy, area))
            H, W = frame.shape[:2]
            area_ratio = float(area) / float(W * H)
            # More conservative U-turn detection - only for very large green areas
            if area_ratio > 0.12:  # Increased threshold from 0.06 to 0.12
                direction = "uturn"
            elif cx < W * 0.35:
                direction = "left"
            elif cx > W * 0.65:
                direction = "right"
            else:
                direction = None
            log(f"Detect_greens fallback: cx={cx} cy={cy} area={area} ratio={area_ratio:.4f} dir={direction}")
            return centroids, direction
        except Exception as e:
            log(f"Erro no fallback de detect_greens: {e}")
            return [], None

    def _infer_curve_direction(self, valids, angle):
        if angle is not None and abs(angle) > 10:
            return "left" if angle < 0 else "right"
        if valids:
            avg_x = sum([p[0] for p in valids]) / len(valids)
            if avg_x < self.WIDTH*0.5:
                return "left"
            else:
                return "right"
        return "left"

def _wire_web(robot: Robot):
    if not WEB_AVAILABLE:
        return
    try:
        if hasattr(web_stream, "register_robot"):
            web_stream.register_robot(robot)
            log("Robô registrado no servidor web.")
    except Exception as e:
        log(f"Falha ao integrar com web_stream: {e}")

def _setup_button(robot: Robot):
    if not GPIO_AVAILABLE:
        log("GPIO indisponível: sem botão físico.")
        return
    for pin in [21, 4]:
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            def _toggle(channel, _pin=pin):
                if GPIO.input(_pin) == GPIO.LOW:
                    if robot.running:
                        robot.stop()
                    else:
                        robot.start()
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=_toggle, bouncetime=300)
            log(f"Botão físico pronto no BCM {pin}.")
            break
        except Exception as e:
            log(f"Falha botão BCM {pin}: {e}")

def main():
    robot = Robot()
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
            robot.start()
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            robot.stop()

if __name__ == "__main__":
    main()
