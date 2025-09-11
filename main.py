#!/usr/bin/env python3
"""
main.py - Robô seguidor de linha (versão robusta)

Principais características:
* Import seguro do RPi.GPIO com fallback para Mock (evita "Cannot determine SOC peripheral base address").
* MotorDriver que usa HardwareControl (se disponível) ou controla pinos diretos + PWM.
* Debounce para sensores (reduz oscilações "vai e volta").
* Start/Stop via botões com debounce; pode iniciar/parar/reiniciar várias vezes.
* Busca controlada quando perde a linha (giro incremental com timeout).
* Suavização (low-pass) dos comandos de motor para evitar mudanças abruptas.
* Logs simples no console.
"""

import time
import threading
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

# ---------- Configurações (ajuste conforme seu robô) ----------
WIDTH = 640
HEIGHT = 480

# Pinos (ajuste conforme seu hardware)
PIN_BTN_START = 21    # botão start
PIN_BTN_STOP = 4      # botão stop
PIN_SENSOR_FRONT = 17
PIN_SENSOR_LEFT = 27
PIN_SENSOR_RIGHT = 22

# pinos motores (usados se HardwareControl NÃO estiver disponível)
PIN_MOTOR_L_FWD = 5
PIN_MOTOR_L_REV = 6
PIN_MOTOR_R_FWD = 13
PIN_MOTOR_R_REV = 19

# parâmetros de controle
BASE_SPEED = 45         # velocidade base (0-100)
MAX_SPEED = 100
KP = 0.025
KD = 0.008
DEAD_ZONE = 8

SENSOR_DEBOUNCE_WINDOW = 3    # número de amostras para debouncing (impar)
MOTOR_SMOOTH_ALPHA = 0.6      # 0..1 (maior = mais inercial, menos alterações bruscas)

SEARCH_TIMEOUT = 1.2          # segundos para busca incremental de linha
SEARCH_STEP = 0.12            # tempo por burst no giro
TURN_BIAS = 110               # intensidade de giro in-place (quando necessário)
# ------------------------------------------------------------

# ----------------- Import seguro do GPIO (com fallback) -----------------
def import_gpio_safe():
    """
    Tenta importar RPi.GPIO e inicializar em BCM.
    Se qualquer etapa falhar (ImportError ou RuntimeError), retorna um MockGPIO.
    """
    try:
        import RPi.GPIO as GPIO_real
        try:
            GPIO_real.setwarnings(False)
            GPIO_real.setmode(GPIO_real.BCM)
            # se chegou aqui, tudo ok
            return GPIO_real, True
        except Exception:
            # falhou ao configurar (ex: não rodando em Pi) -> fallback
            raise
    except Exception:
        # mock simples
        class MockPWM:
            def __init__(self, pin, freq): pass
            def start(self, dc): pass
            def ChangeDutyCycle(self, dc): pass
            def stop(self): pass

        class MockGPIO:
            BCM = "BCM"
            IN = "IN"
            OUT = "OUT"
            LOW = 0
            HIGH = 1
            PUD_UP = "PUD_UP"
            FALLING = "FALLING"
            RISING = "RISING"
            BOTH = "BOTH"

            def setmode(self, *a, **k): pass
            def setwarnings(self, *a, **k): pass
            def setup(self, *a, **k): pass
            def input(self, *a, **k): return self.HIGH
            def output(self, *a, **k): pass
            def add_event_detect(self, *a, **k): pass
            def remove_event_detect(self, *a, **k): pass
            def PWM(self, pin, freq): return MockPWM(pin, freq)
            def cleanup(self): pass

        return MockGPIO(), False

GPIO, _GPIO_VALID = import_gpio_safe()
# -----------------------------------------------------------------------

# ---------- Logger simples ----------
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------- Debouncer para sensores ----------
class Debouncer:
    def __init__(self, gpio_pin: int, active_level=GPIO.LOW, window:int=SENSOR_DEBOUNCE_WINDOW):
        self.pin = gpio_pin
        self.active_level = active_level
        self.window = window if window % 2 == 1 else window+1
        self.buf = deque(maxlen=self.window)
        # inicaliza buffer com não-ativo
        for _ in range(self.window):
            self.buf.append(False)

    def sample(self) -> bool:
        try:
            val = (GPIO.input(self.pin) == self.active_level)
        except Exception:
            # se mock retorna False
            val = False
        self.buf.append(val)
        # maioria das amostras
        return sum(self.buf) >= (self.window // 2 + 1)

    def force_read(self) -> bool:
        """Leitura imediata sem debouncing (uso pontual)."""
        try:
            return (GPIO.input(self.pin) == self.active_level)
        except Exception:
            return False

# ---------- Motor driver que prioriza HardwareControl ----------
class MotorDriver:
    def __init__(self):
        # tenta usar hardware_control se disponível
        self.hw = None
        try:
            from hardware_control import HardwareControl
            self.hw = HardwareControl({"pid": {}})
            log("MotorDriver: usando HardwareControl.")
        except Exception:
            log("MotorDriver: HardwareControl não disponível, usando GPIO direto (ou mock).")
            self.hw = None
            # tenta configurar pinos de motor via GPIO
            try:
                GPIO.setup(PIN_MOTOR_L_FWD, GPIO.OUT)
                GPIO.setup(PIN_MOTOR_L_REV, GPIO.OUT)
                GPIO.setup(PIN_MOTOR_R_FWD, GPIO.OUT)
                GPIO.setup(PIN_MOTOR_R_REV, GPIO.OUT)
                # tenta PWM (se driver fornecer)
                try:
                    self.pwm_l_fwd = GPIO.PWM(PIN_MOTOR_L_FWD, 1000)
                    self.pwm_l_rev = GPIO.PWM(PIN_MOTOR_L_REV, 1000)
                    self.pwm_r_fwd = GPIO.PWM(PIN_MOTOR_R_FWD, 1000)
                    self.pwm_r_rev = GPIO.PWM(PIN_MOTOR_R_REV, 1000)
                    self.pwm_l_fwd.start(0); self.pwm_l_rev.start(0)
                    self.pwm_r_fwd.start(0); self.pwm_r_rev.start(0)
                    self.pwm_enabled = True
                except Exception:
                    # pis digitais apenas
                    self.pwm_enabled = False
                    self.pwm_l_fwd = self.pwm_l_rev = self.pwm_r_fwd = self.pwm_r_rev = None
            except Exception:
                # se GPIO falhar (mock), apenas ignore
                self.pwm_enabled = False
                self.pwm_l_fwd = self.pwm_l_rev = self.pwm_r_fwd = self.pwm_r_rev = None

        # estado para suavização
        self.last_left = 0
        self.last_right = 0

    def _apply_pwm(self, pin_pwm, duty):
        try:
            if pin_pwm is not None:
                pin_pwm.ChangeDutyCycle(max(0.0, min(100.0, abs(duty))))
        except Exception:
            pass

    def drive(self, left: int, right: int):
        """
        left/right: -100 .. 100
        Positivo = para frente, negativo = ré.
        """
        # clipping
        left = int(max(-MAX_SPEED, min(MAX_SPEED, left)))
        right = int(max(-MAX_SPEED, min(MAX_SPEED, right)))

        # suavização (low-pass)
        sm_left = int(self.last_left * MOTOR_SMOOTH_ALPHA + left * (1.0 - MOTOR_SMOOTH_ALPHA))
        sm_right = int(self.last_right * MOTOR_SMOOTH_ALPHA + right * (1.0 - MOTOR_SMOOTH_ALPHA))
        self.last_left, self.last_right = sm_left, sm_right

        if self.hw is not None:
            # HardwareControl geralmente espera dois valores (left,right)
            try:
                if hasattr(self.hw, "drive"):
                    self.hw.drive(sm_left, sm_right)
                    return
                elif hasattr(self.hw, "set_motor_speed"):
                    self.hw.set_motor_speed(sm_left, sm_right)
                    return
            except Exception as e:
                log(f"MotorDriver: falha em HardwareControl.drive: {e}")

        # fallback GPIO direto
        if self.pwm_enabled:
            # esquerda
            if sm_left >= 0:
                self._apply_pwm(self.pwm_l_fwd, sm_left)
                self._apply_pwm(self.pwm_l_rev, 0)
            else:
                self._apply_pwm(self.pwm_l_fwd, 0)
                self._apply_pwm(self.pwm_l_rev, -sm_left)
            # direita
            if sm_right >= 0:
                self._apply_pwm(self.pwm_r_fwd, sm_right)
                self._apply_pwm(self.pwm_r_rev, 0)
            else:
                self._apply_pwm(self.pwm_r_fwd, 0)
                self._apply_pwm(self.pwm_r_rev, -sm_right)
        else:
            # digital on/off: apenas direção baseada em sinal (mantém menos controle de velocidade)
            try:
                # esquerda
                if sm_left > 0:
                    GPIO.output(PIN_MOTOR_L_FWD, GPIO.HIGH); GPIO.output(PIN_MOTOR_L_REV, GPIO.LOW)
                elif sm_left < 0:
                    GPIO.output(PIN_MOTOR_L_FWD, GPIO.LOW); GPIO.output(PIN_MOTOR_L_REV, GPIO.HIGH)
                else:
                    GPIO.output(PIN_MOTOR_L_FWD, GPIO.LOW); GPIO.output(PIN_MOTOR_L_REV, GPIO.LOW)
                # direita
                if sm_right > 0:
                    GPIO.output(PIN_MOTOR_R_FWD, GPIO.HIGH); GPIO.output(PIN_MOTOR_R_REV, GPIO.LOW)
                elif sm_right < 0:
                    GPIO.output(PIN_MOTOR_R_FWD, GPIO.LOW); GPIO.output(PIN_MOTOR_R_REV, GPIO.HIGH)
                else:
                    GPIO.output(PIN_MOTOR_R_FWD, GPIO.LOW); GPIO.output(PIN_MOTOR_R_REV, GPIO.LOW)
            except Exception:
                pass

    def stop(self):
        # parar via hw ou GPIO
        try:
            if self.hw is not None and hasattr(self.hw, "stop"):
                self.hw.stop()
        except Exception:
            pass
        # PWM ou digital -> zerar
        try:
            if self.pwm_enabled:
                self._apply_pwm(self.pwm_l_fwd, 0)
                self._apply_pwm(self.pwm_l_rev, 0)
                self._apply_pwm(self.pwm_r_fwd, 0)
                self._apply_pwm(self.pwm_r_rev, 0)
            else:
                GPIO.output(PIN_MOTOR_L_FWD, GPIO.LOW)
                GPIO.output(PIN_MOTOR_L_REV, GPIO.LOW)
                GPIO.output(PIN_MOTOR_R_FWD, GPIO.LOW)
                GPIO.output(PIN_MOTOR_R_REV, GPIO.LOW)
        except Exception:
            pass
# ---------------------------------------------------------------------

# ---------- Visão mínima (binarização + centroid por strip) ----------
def binarize(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # ROI inferior para reduzir ruído
    h = frame_bgr.shape[0]
    roi = gray[h-140:h, :]
    blur = cv2.GaussianBlur(roi, (5,5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    # devolve imagem binária com mesma largura, mas reduzida em altura (atenção índices)
    full_bw = np.zeros_like(gray)
    full_bw[h-140:h, :] = bw
    return full_bw

def strip_centroid(bw, y0, h):
    H, W = bw.shape[:2]
    y0 = max(0, min(H-1, int(y0)))
    y1 = max(0, min(H, int(y0 + h)))
    roi = bw[y0:y1, :]
    colsum = roi.sum(axis=0)
    threshold = 255 * (y1-y0) * 0.03
    if colsum.max() < threshold:
        return None, 0
    x = np.arange(0, W, dtype=np.float32)
    cx = float((x * colsum).sum() / (colsum.sum() + 1e-6))
    cy = int((y0 + y1)/2)
    return (int(cx), int(cy)), int(np.count_nonzero(colsum))

# ---------- PD Controller simplificado ----------
class PDController:
    def __init__(self, kp=KP, kd=KD, base_speed=BASE_SPEED, max_speed=MAX_SPEED, dead_zone=DEAD_ZONE):
        self.kp = kp
        self.kd = kd
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.dead_zone = dead_zone
        self.last_error = 0.0

    def compute(self, centroid: Optional[Tuple[int,int]], width: int):
        if centroid is None:
            return None  # indica perda da linha
        err = centroid[0] - (width // 2)
        derivative = err - self.last_error
        self.last_error = err
        corr = int(self.kp * err + self.kd * derivative)
        # velocidade adaptativa: reduz em curvas
        adaptive = max(20, int(self.base_speed - abs(err)*0.04))
        left = adaptive - corr
        right = adaptive + corr
        if abs(err) < self.dead_zone:
            left = right = adaptive
        left = int(np.clip(left, -self.max_speed, self.max_speed))
        right = int(np.clip(right, -self.max_speed, self.max_speed))
        return left, right

# ---------- Robot (coordena visão, sensores e motores) ----------
class Robot:
    def __init__(self):
        # câmera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        # drivers & control
        self.motor = MotorDriver()
        self.pd = PDController()
        # sensores com debouncer (assumindo circuito pull-up -> LOW = ativo)
        try:
            GPIO.setup(PIN_SENSOR_FRONT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(PIN_SENSOR_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(PIN_SENSOR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        except Exception:
            # se mock, ignore
            pass
        self.s_front = Debouncer(PIN_SENSOR_FRONT, active_level=GPIO.LOW)
        self.s_left = Debouncer(PIN_SENSOR_LEFT, active_level=GPIO.LOW)
        self.s_right = Debouncer(PIN_SENSOR_RIGHT, active_level=GPIO.LOW)

        # start/stop management
        self._lock = threading.Lock()
        self.running = False
        self._thread = None
        # ultimo comando aplicado (para info)
        self.last_cmd = (0,0)

    def start(self):
        with self._lock:
            if self.running:
                log("Robot.start(): já está rodando.")
                return
            self.running = True
            self._thread = threading.Thread(target=self._main_loop, daemon=True)
            self._thread.start()
            log("Robot iniciado.")

    def stop(self):
        with self._lock:
            if not self.running:
                log("Robot.stop(): já está parado.")
                return
            self.running = False
        # espera o thread encerrar por pequeno tempo
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.motor.stop()
        except Exception:
            pass
        log("Robot parado.")

    def toggle(self):
        with self._lock:
            cur = self.running
        if cur:
            self.stop()
        else:
            self.start()

    def _turn_until_line(self, direction: str, timeout_s: float) -> bool:
        """
        Gira incrementalmente até achar linha nos strips inferiores.
        direction: 'left' ou 'right'.
        Retorna True se encontrou, False se timeout.
        """
        sign = 1 if direction == "left" else -1
        base_bias = TURN_BIAS
        max_steps = max(1, int(timeout_s / SEARCH_STEP))
        start_time = time.time()
        found = False
        try:
            for step in range(max_steps):
                ramp = 0.5 + 0.5 * (step / max_steps)
                bias = int(sign * base_bias * ramp)
                burst_start = time.time()
                # executar burst
                while (time.time() - burst_start) < SEARCH_STEP and (time.time() - start_time) < timeout_s and self.running:
                    # comando de giro in-place: left = -bias/2, right = bias/2 (sinalizado)
                    self.motor.drive(int(-bias/2), int(bias/2))
                    # verifica frame
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    bw = binarize(frame)
                    # checar dois strips inferiores
                    for k in range(2):
                        y0 = HEIGHT - 20 - k*28
                        c, w = strip_centroid(bw, y0, 28)
                        if c is not None and w > 0:
                            found = True
                            break
                    if found:
                        break
                    time.sleep(0.02)
                if found or not self.running:
                    break
            # anti-stuck simples: se não encontrou, dar pequeno burst oposto
            if not found and self.running:
                opp_bias = int(-sign * base_bias * 0.6)
                t0 = time.time()
                while time.time() - t0 < 0.35 and self.running:
                    self.motor.drive(int(-opp_bias/2), int(opp_bias/2))
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.02); continue
                    bw = binarize(frame)
                    for k in range(2):
                        y0 = HEIGHT - 20 - k*28
                        c, w = strip_centroid(bw, y0, 28)
                        if c is not None and w > 0:
                            found = True
                            break
                    if found:
                        break
                    time.sleep(0.02)
        finally:
            self.motor.stop()
        if found:
            # pequeno avanço para estabilizar
            self.motor.drive(int(BASE_SPEED*0.6), int(BASE_SPEED*0.6))
            time.sleep(0.12)
            self.motor.stop()
            return True
        return False

    def _main_loop(self):
        log("Loop principal do robô iniciado.")
        try:
            while True:
                with self._lock:
                    if not self.running:
                        break
                # sample sensores (debounced)
                front = self.s_front.sample()
                left_s = self.s_left.sample()
                right_s = self.s_right.sample()

                # prioridade obstáculo frontal
                if front:
                    log("Obstáculo detectado (frontal). Fazendo manobra de evasão.")
                    # movimento controlado: ré curto + giro + stop
                    self.motor.drive(-40, -40)
                    time.sleep(0.28)
                    self.motor.stop()
                    time.sleep(0.07)
                    # girar à direita livremente por curto tempo
                    self.motor.drive(-TURN_BIAS//3, TURN_BIAS//3)
                    time.sleep(0.32)
                    self.motor.stop()
                    # pular pro início do loop -> continuar
                    time.sleep(0.05)
                    continue

                # leitura câmera
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                bw = binarize(frame)

                # tenta obter centroid do primeiro strip inferior
                y0 = HEIGHT - 20
                centroid, width = strip_centroid(bw, y0, 28)
                pd_out = self.pd.compute(centroid, WIDTH)

                if pd_out is None:
                    # linha perdida -> busca incremental (primeira tentativa à esquerda, depois direita)
                    log("Linha perdida -> iniciando busca")
                    found = self._turn_until_line("left", timeout_s=SEARCH_TIMEOUT)
                    if not found:
                        found = self._turn_until_line("right", timeout_s=SEARCH_TIMEOUT)
                    if not found:
                        # anti-stuck curto: avanço pequeno e para
                        log("Busca falhou -> anti-stuck (avançar curto)")
                        self.motor.drive(int(BASE_SPEED*0.4), int(BASE_SPEED*0.4))
                        time.sleep(0.16)
                        self.motor.stop()
                    continue
                else:
                    left_cmd, right_cmd = pd_out
                    # garantir limites e enviar para motor
                    self.motor.drive(left_cmd, right_cmd)
                    self.last_cmd = (left_cmd, right_cmd)

                # pequeno sleep para aliviar CPU
                time.sleep(0.01)
        except Exception as e:
            log(f"Erro no loop do robô: {e}")
        finally:
            try:
                self.motor.stop()
            except Exception:
                pass
            log("Loop principal finalizado.")

# ---------- Setup de botões (start/stop) com debounce ----------
def setup_buttons(robot: Robot):
    # configura pinos dos botões se possível
    try:
        GPIO.setup(PIN_BTN_START, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PIN_BTN_STOP, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    except Exception:
        pass

    # callback simples (gera toggle/start/stop)
    def cb_start(channel=None):
        # callback curto para iniciar (chamado em FALLING)
        log("Botão START pressionado -> start()")
        try:
            robot.start()
        except Exception as e:
            log(f"Erro ao executar start(): {e}")

    def cb_stop(channel=None):
        log("Botão STOP pressionado -> stop()")
        try:
            robot.stop()
        except Exception as e:
            log(f"Erro ao executar stop(): {e}")

    # registra event detect com bouncetime se disponível
    try:
        GPIO.add_event_detect(PIN_BTN_START, GPIO.FALLING, callback=cb_start, bouncetime=300)
        GPIO.add_event_detect(PIN_BTN_STOP, GPIO.FALLING, callback=cb_stop, bouncetime=300)
        log("Botões configurados com event detect.")
    except Exception:
        # fallback: sem event detect (mock), usuário poderá usar teclado
        log("add_event_detect não disponível; usando polling (fallback).")
        # cria thread de polling
        def poll_buttons():
            last_start = GPIO.HIGH
            last_stop = GPIO.HIGH
            while True:
                try:
                    cur_start = GPIO.input(PIN_BTN_START)
                    cur_stop = GPIO.input(PIN_BTN_STOP)
                except Exception:
                    cur_start = GPIO.HIGH; cur_stop = GPIO.HIGH
                if cur_start == GPIO.LOW and last_start == GPIO.HIGH:
                    cb_start()
                if cur_stop == GPIO.LOW and last_stop == GPIO.HIGH:
                    cb_stop()
                last_start, last_stop = cur_start, cur_stop
                time.sleep(0.08)
        t = threading.Thread(target=poll_buttons, daemon=True)
        t.start()

# ---------- main ----------
def main():
    log("Iniciando main.py")
    # instanciar robô
    robot = Robot()
    setup_buttons(robot)

    # interface mínima no terminal: permite start/stop via teclado também
    log("Pressione 's' + Enter para START, 'x' + Enter para STOP, Ctrl+C para sair.")
    try:
        while True:
            # print status
            with robot._lock:
                running = robot.running
            status = "RUNNING" if running else "STOPPED"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status} - last_cmd={robot.last_cmd}", end="\r")
            # leitura não bloqueante do teclado: usar input com timeout simples
            # implementado via sleep curto (buttons via GPIO callbacks já fazem start/stop)
            time.sleep(0.6)
    except KeyboardInterrupt:
        log("KeyboardInterrupt recebido. Encerrando.")
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        log("main.py finalizado.")

if __name__ == "__main__":
    main()
