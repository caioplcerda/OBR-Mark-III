# hardware_control.py
# Backend duplo: RPi.GPIO (padrão) com fallback automático para lgpio (Pi 5 / Bookworm).
# Mantém a mesma API esperada pelo main.py:
# - set_motor_speed(base_speed, error) com PID direcional interno
# - Encoders (se presentes) para malha fechada de velocidade
# - 4 servos com posições A/B
# Pinos: TB6612FNG canal B (L: 17/27/22; R: 23/24/25) + STBY=5 (ligado em HW)

import time
import threading

# =========================
#   BACKEND GPIO ABSTRATO
# =========================
class _PWMBase:
    def start(self, duty): raise NotImplementedError
    def ChangeDutyCycle(self, duty): raise NotImplementedError
    def stop(self): raise NotImplementedError

class _GPIOBackendBase:
    OUT = 1
    IN = 0
    PUD_UP = 2
    RISING = 31

    def setup(self, pin, mode, pull_up_down=None): raise NotImplementedError
    def output(self, pin, level): raise NotImplementedError
    def input(self, pin): raise NotImplementedError
    def PWM(self, pin, freq_hz): raise NotImplementedError
    def add_event_detect(self, pin, edge, callback): raise NotImplementedError
    def remove_event_detect(self, pin): pass
    def cleanup(self): pass

# ---------- Backend 1: RPi.GPIO ----------
class _RPiGPIOBackend(_GPIOBackendBase):
    def __init__(self):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        # Pode disparar RuntimeError no Pi 5 (base address). Vamos deixar try/except no caller.
        self._GPIO.setwarnings(False)
        self._GPIO.setmode(self._GPIO.BCM)
        # map
        self.OUT = self._GPIO.OUT
        self.IN = self._GPIO.IN
        self.PUD_UP = self._GPIO.PUD_UP
        self.RISING = self._GPIO.RISING

    def setup(self, pin, mode, pull_up_down=None):
        if pull_up_down is None:
            self._GPIO.setup(pin, mode)
        else:
            self._GPIO.setup(pin, mode, pull_up_down=pull_up_down)

    def output(self, pin, level):
        self._GPIO.output(pin, self._GPIO.HIGH if level else self._GPIO.LOW)

    def input(self, pin):
        return self._GPIO.input(pin)

    class _PWM(_PWMBase):
        def __init__(self, GPIO, pin, freq):
            self._pwm = GPIO.PWM(pin, freq)
        def start(self, duty): self._pwm.start(max(0.0, min(100.0, float(duty))))
        def ChangeDutyCycle(self, duty): self._pwm.ChangeDutyCycle(max(0.0, min(100.0, float(duty))))
        def stop(self): 
            try: self._pwm.stop()
            except Exception: pass

    def PWM(self, pin, freq_hz):
        return self._PWM(self._GPIO, pin, freq_hz)

    def add_event_detect(self, pin, edge, callback):
        self._GPIO.add_event_detect(pin, edge, callback=callback)

    def remove_event_detect(self, pin):
        try: self._GPIO.remove_event_detect(pin)
        except Exception: pass

    def cleanup(self):
        try: self._GPIO.cleanup()
        except Exception: pass

# ---------- Backend 2: lgpio ----------
class _LGPIOBackend(_GPIOBackendBase):
    def __init__(self, chip_index=0):
        import lgpio
        self._lgpio = lgpio
        self._h = self._lgpio.gpiochip_open(chip_index)
        # constants
        self.OUT = 1
        self.IN = 0
        self.PUD_UP = 2
        self.RISING = self._lgpio.RISING_EDGE
        # keep track of claimed pins
        self._claimed = set()
        self._alerts = {}

    def _claim_out(self, pin, init=0):
        if pin not in self._claimed:
            self._lgpio.gpio_claim_output(self._h, pin, init)
            self._claimed.add(pin)

    def _claim_in(self, pin, pull_up_down=None):
        if pin not in self._claimed:
            flags = 0
            if pull_up_down == self.PUD_UP:
                flags |= self._lgpio.SET_PULL_UP
            self._lgpio.gpio_claim_input(self._h, pin, flags)
            self._claimed.add(pin)

    def setup(self, pin, mode, pull_up_down=None):
        if mode == self.OUT:
            self._claim_out(pin, 0)
        else:
            self._claim_in(pin, pull_up_down)

    def output(self, pin, level):
        self._claim_out(pin, 0)
        self._lgpio.gpio_write(self._h, pin, 1 if level else 0)

    def input(self, pin):
        self._claim_in(pin, self.PUD_UP)
        return self._lgpio.gpio_read(self._h, pin)

    class _PWM(_PWMBase):
        def __init__(self, lgpio_mod, handle, pin, freq):
            self._lgpio = lgpio_mod
            self._h = handle
            self._pin = pin
            self._freq = int(freq)
            self._duty = 0.0
            # claim output
            try: self._lgpio.gpio_claim_output(self._h, pin, 0)
            except Exception: pass
        def start(self, duty):
            self.ChangeDutyCycle(duty)
        def ChangeDutyCycle(self, duty):
            self._duty = max(0.0, min(100.0, float(duty)))
            self._lgpio.tx_pwm(self._h, self._pin, self._freq, self._duty)
        def stop(self):
            try: self._lgpio.tx_pwm(self._h, self._pin, self._freq, 0)
            except Exception: pass

    def PWM(self, pin, freq_hz):
        return self._PWM(self._lgpio, self._h, pin, freq_hz)

    def add_event_detect(self, pin, edge, callback):
        # Configura alerta de borda no lgpio
        self._claim_in(pin, self.PUD_UP)
        self._lgpio.gpio_claim_alert(self._h, pin, self.RISING, 0)
        # Registra callback
        def _cb(_chip, _gpio, level, _tick):
            # RISING -> level == 1 (mas alguns firmwares usam 0/1/2)
            try:
                if level == 1 or level == self._lgpio.RISING_EDGE:
                    callback(pin)
            except Exception:
                pass
        self._alerts[pin] = _cb
        self._lgpio.set_alert_func(self._h, pin, _cb)

    def remove_event_detect(self, pin):
        if pin in self._alerts:
            try:
                self._lgpio.set_alert_func(self._h, pin, None)
            except Exception:
                pass
            self._alerts.pop(pin, None)

    def cleanup(self):
        try:
            for pin in list(self._alerts.keys()):
                self.remove_event_detect(pin)
        except Exception:
            pass
        try:
            self._lgpio.gpiochip_close(self._h)
        except Exception:
            pass

# Seleção de backend
def _get_gpio_backend():
    # 1) tenta RPi.GPIO
    try:
        return _RPiGPIOBackend(), "rpi"
    except RuntimeError as e:
        msg = str(e)
        if "Cannot determine SOC peripheral base address" in msg:
            # Pi 5 / base antiga: força lgpio
            try:
                return _LGPIOBackend(), "lgpio"
            except Exception as ee:
                raise RuntimeError(f"Falha no backend lgpio: {ee}") from e
        else:
            raise
    except Exception as e:
        # fallback lgpio
        try:
            return _LGPIOBackend(), "lgpio"
        except Exception as ee:
            raise RuntimeError(f"Falha ao inicializar GPIO: {e} / lgpio: {ee}")

# =========================
#   PID genérico
# =========================
class PIDController:
    def __init__(self, kp=0.4, ki=0.0, kd=0.1, setpoint=0.0, sample_time=0.01, out_min=None, out_max=None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.sample_time = float(sample_time)
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_t = time.time()
        self.out_min = out_min
        self.out_max = out_max

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_t = time.time()

    def calculate(self, measurement):
        now = time.time()
        dt = now - self._last_t
        if dt < self.sample_time:
            return self._prev_error  # saída “estável” entre amostras

        error = self.setpoint - float(measurement)
        self._integral += error * dt
        # anti-windup simples
        self._integral = max(-200.0, min(200.0, self._integral))
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0

        out = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        if self.out_min is not None: out = max(self.out_min, out)
        if self.out_max is not None: out = min(self.out_max, out)

        self._prev_error = error
        self._last_t = now
        return out

# =========================
#   HardwareControl
# =========================
class HardwareControl:
    """
    - TB6612FNG (canal B)
    - Encoders em loop fechado de velocidade (se disponíveis)
    - API: set_motor_speed(base_speed, error)
    - 4 servos com duas posições (A/B)
    """

    # ======= CONFIG MOTOR/ENCODER =======
    TICKS_PER_REV = 20           # ajuste para seu encoder
    MAX_TICKS_PER_SEC = 300.0    # pico estimado em 100% PWM (ajuste na calibração)
    CONTROL_HZ = 50              # Hz do laço de velocidade

    # PID de velocidade por roda (saída = ajuste PWM)
    VEL_PID_L = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)
    VEL_PID_R = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)

    # ======= SERVOS =======
    SERVO_PINS = [7, 18, 16, 20]   # 12 está nos LEDs WS2812; usamos estes 4
    SERVO_FREQ_HZ = 50
    SERVO_AB_US = [
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
    ]

    # Pinos TB6612FNG (canal B)
    L_BIN1, L_BIN2, L_PWMB = 17, 27, 22
    R_BIN1, R_BIN2, R_PWMB = 23, 24, 25
    STBY = 5  # ligado em HIGH no hardware

    # Encoders (canal A com interrupção)
    ENCODER_A_L, ENCODER_B_L = 6, 13
    ENCODER_A_R, ENCODER_B_R = 19, 26

    def __init__(self, config):
        self.config = config or {}

        # --- escolhe backend ---
        self.GPIO, self.backend = _get_gpio_backend()
        # print(f"[HardwareControl] Backend GPIO: {self.backend}")

        # Estados dos encoders
        self._ticks_l = 0
        self._ticks_r = 0

        # Medidas (ticks/s)
        self.meas_tps_l = 0.0
        self.meas_tps_r = 0.0

        # Alvos (ticks/s)
        self.target_tps_l = 0.0
        self.target_tps_r = 0.0

        # Telemetria
        self.last_left_speed = 0   # duty/sinal final [-100..100]
        self.last_right_speed = 0

        # Controle: se não houver encoder, operamos em malha aberta
        self._encoders_ok = True

        # ---- GPIO: motores ----
        for pin in [self.L_BIN1, self.L_BIN2, self.R_BIN1, self.R_BIN2, self.STBY]:
            self.GPIO.setup(pin, self.GPIO.OUT)
        self.GPIO.setup(self.L_PWMB, self.GPIO.OUT)
        self.GPIO.setup(self.R_PWMB, self.GPIO.OUT)

        # PWM motores (1 kHz para suavidade)
        self._pwm_left = self.GPIO.PWM(self.L_PWMB, 1000)
        self._pwm_right = self.GPIO.PWM(self.R_PWMB, 1000)
        self._pwm_left.start(0)
        self._pwm_right.start(0)

        # STBY: mantemos HIGH (você ligou em HW)
        self.GPIO.output(self.STBY, 1)

        # ---- Encoders ----
        try:
            self.GPIO.setup(self.ENCODER_A_L, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP)
            self.GPIO.setup(self.ENCODER_A_R, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP)
            self.GPIO.add_event_detect(self.ENCODER_A_L, self.GPIO.RISING, callback=self._enc_l)
            self.GPIO.add_event_detect(self.ENCODER_A_R, self.GPIO.RISING, callback=self._enc_r)
            self._encoders_ok = True
        except Exception:
            self._encoders_ok = False  # sem encoders conectados

        # PID “direcional” (mistura base/erro vindo do main)
        self.dir_pid = PIDController(
            kp=self.config.get("pid", {}).get("kp", 0.9),
            ki=self.config.get("pid", {}).get("ki", 0.0),
            kd=self.config.get("pid", {}).get("kd", 0.14),
            sample_time=self.config.get("pid", {}).get("sample_time", 0.02)
        )

        # PIDs de velocidade por roda
        self.pid_vel_l = PIDController(out_min=-100.0, out_max=100.0, **self.VEL_PID_L)
        self.pid_vel_r = PIDController(out_min=-100.0, out_max=100.0, **self.VEL_PID_R)

        # Servos
        self._servo_pwm = [None, None, None, None]
        self._servo_ready = [False, False, False, False]
        self._init_servos()

        # ====== IMPORTANTE: inicializar medidores ANTES da thread ======
        self._last_meas_t = time.time()
        self._last_ticks_l = 0
        self._last_ticks_r = 0

        # Thread de controle de velocidade
        self._ctrl_running = True
        self._ctrl_th = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_th.start()

    # --------------------- Encoders callbacks ---------------------
    def _enc_l(self, _):
        self._ticks_l += 1

    def _enc_r(self, _):
        self._ticks_r += 1

    # --------------------- API pública encoders -------------------
    def reset_encoders(self):
        self._ticks_l = 0
        self._ticks_r = 0
        self._last_ticks_l = 0
        self._last_ticks_r = 0
        self.meas_tps_l = 0.0
        self.meas_tps_r = 0.0

    def read_encoders(self):
        return self._ticks_l, self._ticks_r

    def read_speeds_tps(self):
        """Velocidade medida (ticks/s) de cada roda (filtrada)."""
        return self.meas_tps_l, self.meas_tps_r

    # --------------------- Motores (API esperada pelo main) -------
    def set_motor_speed(self, base_speed, error):
        """
        Recebe (base, erro) do controlador de linha.
        Converte em comandos por roda (-100..100) e define alvos de velocidade (ticks/s).
        Laço interno (_control_loop) usa encoders p/ ajustar PWM; se não houver encoder, opera em malha aberta.
        """
        correction = self.dir_pid.calculate(error)

        cmd_left = float(base_speed) - float(correction)
        cmd_right = float(base_speed) + float(correction)

        # Limita
        cmd_left = max(-100.0, min(100.0, cmd_left))
        cmd_right = max(-100.0, min(100.0, cmd_right))

        # Guarda p/ UI
        self.last_left_speed = int(cmd_left)
        self.last_right_speed = int(cmd_right)

        if self._encoders_ok:
            # Converte para alvos de velocidade (ticks/s)
            self.target_tps_l = (cmd_left / 100.0) * self.MAX_TICKS_PER_SEC
            self.target_tps_r = (cmd_right / 100.0) * self.MAX_TICKS_PER_SEC
        else:
            # Sem encoders → escreve PWM direto
            self._write_motor_pwm(cmd_left, cmd_right)

    def stop(self):
        # zera alvos/pwm; mantém STBY como está
        self.target_tps_l = 0.0
        self.target_tps_r = 0.0
        self._write_motor_pwm(0.0, 0.0)

    # --------------------- Laço de velocidade por roda ------------
    def _control_loop(self):
        """
        Roda em ~CONTROL_HZ:
        - Calcula velocidade medida (ticks/s)
        - PID de velocidade ajusta PWM por roda
        """
        period = 1.0 / float(self.CONTROL_HZ)
        pwm_l = 0.0
        pwm_r = 0.0
        alpha = 0.4  # EMA

        while self._ctrl_running:
            t0 = time.time()

            if self._encoders_ok:
                # Mede ticks/s
                now = time.time()
                dt = now - self._last_meas_t
                if dt <= 0:
                    dt = period

                ticks_l = self._ticks_l
                ticks_r = self._ticks_r
                dticks_l = ticks_l - self._last_ticks_l
                dticks_r = ticks_r - self._last_ticks_r

                inst_tps_l = float(dticks_l) / dt
                inst_tps_r = float(dticks_r) / dt

                # EMA
                self.meas_tps_l = alpha * inst_tps_l + (1 - alpha) * self.meas_tps_l
                self.meas_tps_r = alpha * inst_tps_r + (1 - alpha) * self.meas_tps_r

                self._last_ticks_l = ticks_l
                self._last_ticks_r = ticks_r
                self._last_meas_t = now

                # PIDs de velocidade
                self.pid_vel_l.setpoint = self.target_tps_l
                self.pid_vel_r.setpoint = self.target_tps_r

                adj_l = self.pid_vel_l.calculate(self.meas_tps_l)  # –100..100 (limitado no PID)
                adj_r = self.pid_vel_r.calculate(self.meas_tps_r)

                # Atualiza PWM suavemente
                pwm_l = max(-100.0, min(100.0, pwm_l + adj_l * 0.2))
                pwm_r = max(-100.0, min(100.0, pwm_r + adj_r * 0.2))

                # Escreve
                self._write_motor_pwm(pwm_l, pwm_r)
            else:
                # Sem encoders → já está sendo escrito em set_motor_speed
                pass

            # Dorme até completar o período
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _write_motor_pwm(self, left_cmd, right_cmd):
        """
        left_cmd/right_cmd ∈ [-100..100] → define sentido (BIN1/BIN2) e duty.
        Observação: lado direito não invertido aqui; ajuste conforme seu teste.
        """
        # Esquerda
        if left_cmd >= 0:
            self.GPIO.output(self.L_BIN1, 1)
            self.GPIO.output(self.L_BIN2, 0)
        else:
            self.GPIO.output(self.L_BIN1, 0)
            self.GPIO.output(self.L_BIN2, 1)
        self._pwm_left.ChangeDutyCycle(abs(left_cmd))

        # Direita (ajuste conforme seu teste_v2)
        if right_cmd >= 0:
            self.GPIO.output(self.R_BIN1, 1)
            self.GPIO.output(self.R_BIN2, 0)
        else:
            self.GPIO.output(self.R_BIN1, 0)
            self.GPIO.output(self.R_BIN2, 1)
        self._pwm_right.ChangeDutyCycle(abs(right_cmd))

    # --------------------- Servos -------------------------
    @staticmethod
    def _us_to_duty(us, period_ms=20.0):
        return max(0.0, min(100.0, (us / 1000.0) / period_ms * 100.0))

    def _servo_write_us(self, index, us):
        i = index - 1
        if not (0 <= i < 4) or not self._servo_ready[i]:
            return False
        duty = self._us_to_duty(us)
        self._servo_pwm[i].ChangeDutyCycle(duty)
        return True

    def set_servo(self, index, pos, smooth_ms=0):
        i = index - 1
        if not (0 <= i < 4) or pos not in ('A', 'B'):
            return False
        target_us = int(self.SERVO_AB_US[i][pos])
        if smooth_ms and smooth_ms > 0:
            other = 'A' if pos == 'B' else 'B'
            start_us = int(self.SERVO_AB_US[i][other])
            steps = max(3, int(smooth_ms / 20))
            for t in range(steps + 1):
                u = int(start_us + (target_us - start_us) * (t / steps))
                self._servo_write_us(index, u)
                time.sleep(0.02)
        else:
            self._servo_write_us(index, target_us)
        return True

    def set_servos(self, positions=('A', 'A', 'A', 'A'), smooth_ms=0):
        for idx, p in enumerate(positions, start=1):
            self.set_servo(idx, p, smooth_ms=smooth_ms)

    def set_servo_us(self, index, us):
        return self._servo_write_us(index, int(us))

    def calibrate_servo(self, index, A_us=None, B_us=None, commit=True):
        i = index - 1
        if not (0 <= i < 4):
            return False
        if A_us is not None:
            self.SERVO_AB_US[i]['A'] = int(A_us)
        if B_us is not None:
            self.SERVO_AB_US[i]['B'] = int(B_us)
        return True

    def release_servos(self):
        for pwm in self._servo_pwm:
            if pwm:
                pwm.ChangeDutyCycle(0)

    def _init_servos(self):
        # Inicia PWM de 50 Hz nos 4 pinos
        self._servo_pwm = [None, None, None, None]
        self._servo_ready = [False, False, False, False]
        for i, pin in enumerate(self.SERVO_PINS):
            self.GPIO.setup(pin, self.GPIO.OUT)
            pwm = self.GPIO.PWM(pin, self.SERVO_FREQ_HZ)
            pwm.start(0)
            self._servo_pwm[i] = pwm
            self._servo_ready[i] = True
            try:
                self.set_servo(i+1, 'A', smooth_ms=200)
            except Exception:
                pass
        time.sleep(0.1)

    # --------------------- Cleanup ------------------------
    def cleanup(self):
        # para thread
        self._ctrl_running = False
        try:
            if hasattr(self, "_ctrl_th") and self._ctrl_th and self._ctrl_th.is_alive():
                self._ctrl_th.join(timeout=0.5)
        except Exception:
            pass

        # para motores e servos
        try:
            self.stop()
        except Exception:
            pass

        try:
            for pwm in self._servo_pwm:
                if pwm:
                    pwm.stop()
        except Exception:
            pass

        try:
            self._pwm_left.stop()
            self._pwm_right.stop()
        except Exception:
            pass

        try:
            self.GPIO.cleanup()
        except Exception:
            pass
