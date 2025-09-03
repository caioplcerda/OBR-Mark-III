# hardware_control.py
# Open-loop por padrão (reto robusto) e backend forçado em lgpio (PWM por hardware).
# Ajustes:
# - Direita invertida (chassi espelhado): INVERT_RIGHT = True
# - Deadband 55% para arrancada.
# - Snap-to-straight (erro pequeno => 80/80).
# - Encoders opcionais. TICKS_PER_REV = 36.
# - PWM 500 Hz (mais torque).
# - Sem dependência de RPi.GPIO (mas mantém fallback helpers).

import time
import threading

# =========================
#   Backends GPIO (RPi.GPIO -> lgpio; aqui forçamos lgpio)
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

class _LGPIOBackend(_GPIOBackendBase):
    def __init__(self, chip_index=0):
        import lgpio
        self._lgpio = lgpio
        self._h = self._lgpio.gpiochip_open(chip_index)
        self.OUT = 1; self.IN = 0; self.PUD_UP = 2
        self.RISING = self._lgpio.RISING_EDGE
        self._alerts = {}
    def _claim_out(self, pin, init=0):
        self._lgpio.gpio_claim_output(self._h, pin, init)
    def _claim_in(self, pin, pud=None):
        flags = 0
        if pud == self.PUD_UP:
            flags |= self._lgpio.SET_PULL_UP
        self._lgpio.gpio_claim_input(self._h, pin, flags)
    def setup(self, pin, mode, pull_up_down=None):
        if mode == self.OUT: self._claim_out(pin, 0)
        else: self._claim_in(pin, pull_up_down)
    def output(self, pin, level):
        self._lgpio.gpio_write(self._h, pin, 1 if level else 0)
    def input(self, pin):
        return self._lgpio.gpio_read(self._h, pin)
    class _PWM(_PWMBase):
        def __init__(self, lgpio_mod, handle, pin, freq):
            self._lgpio = lgpio_mod; self._h = handle; self._pin = pin; self._freq = int(freq)
            self._lgpio.gpio_claim_output(self._h, pin, 0)
        def start(self, duty):
            self.ChangeDutyCycle(duty)
        def ChangeDutyCycle(self, duty):
            self._lgpio.tx_pwm(self._h, self._pin, self._freq, max(0.0, min(100.0, float(duty))))
        def stop(self):
            try: self._lgpio.tx_pwm(self._h, self._pin, self._freq, 0)
            except Exception: pass
    def PWM(self, pin, freq_hz):
        return self._PWM(self._lgpio, self._h, pin, freq_hz)
    def add_event_detect(self, pin, edge, callback):
        self._lgpio.gpio_claim_input(self._h, pin, self._lgpio.SET_PULL_UP)
        self._lgpio.gpio_claim_alert(self._h, pin, self.RISING, 0)
        def _cb(_chip, _gpio, level, _tick):
            try:
                if level == 1:
                    callback(pin)
            except Exception:
                pass
        self._alerts[pin] = _cb
        self._lgpio.set_alert_func(self._h, pin, _cb)
    def remove_event_detect(self, pin):
        if pin in self._alerts:
            try: self._lgpio.set_alert_func(self._h, pin, None)
            except Exception: pass
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

def _make_backend_forced_lgpio():
    return _LGPIOBackend(), "lgpio"

# =========================
#   PID direcional (para mix base/erro)
# =========================
class PIDController:
    def __init__(self, kp=0.4, ki=0.0, kd=0.1, setpoint=0.0, sample_time=0.01, out_min=None, out_max=None):
        self.kp=float(kp); self.ki=float(ki); self.kd=float(kd)
        self.setpoint=float(setpoint); self.sample_time=float(sample_time)
        self._prev_error=0.0; self._integral=0.0; self._last_t=time.time()
        self.out_min=out_min; self.out_max=out_max
    def reset(self):
        self._prev_error=0.0; self._integral=0.0; self._last_t=time.time()
    def calculate(self, measurement):
        now=time.time(); dt=now-self._last_t
        if dt<self.sample_time: return self._prev_error
        error=self.setpoint-float(measurement)
        self._integral = max(-200.0, min(200.0, self._integral + error*dt))
        derivative=(error-self._prev_error)/dt if dt>0 else 0.0
        out=(self.kp*error)+(self.ki*self._integral)+(self.kd*derivative)
        if self.out_min is not None: out=max(self.out_min,out)
        if self.out_max is not None: out=min(self.out_max,out)
        self._prev_error=error; self._last_t=now
        return out

# =========================
#   HardwareControl
# =========================
class HardwareControl:
    """
    Open-loop por padrão (reto robusto). Encoders opcionais.
    """

    # ---- Ajustes principais ----
    MIN_DUTY = 55.0          # deadband para vencer inércia
    OPEN_LOOP = True         # por padrão, PWM direto (encoder não fecha malha)
    AUTO_FALLBACK_OPEN = True
    FALLBACK_CHECK_SEC = 0.7

    # Snap-to-straight: quando o erro for pequeno, força 80/80
    STRAIGHT_ERR_TH = 12.0
    STRAIGHT_MATCH_DELTA = 10.0
    STRAIGHT_MIN_DUTY = 80.0

    # Encoders
    TICKS_PER_REV = 36
    MAX_TICKS_PER_SEC = 300.0
    CONTROL_HZ = 50
    VEL_PID_L = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)
    VEL_PID_R = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)

    # Servos
    SERVO_PINS = [7, 18, 16, 20]
    SERVO_FREQ_HZ = 50
    SERVO_AB_US = [{"A":1000,"B":2000} for _ in range(4)]

    # TB6612 canal B
    L_BIN1, L_BIN2, L_PWMB = 17, 27, 22
    R_BIN1, R_BIN2, R_PWMB = 23, 24, 25
    STBY = 5

    # Encoders (canal A)
    ENCODER_A_L, ENCODER_B_L = 6, 13
    ENCODER_A_R, ENCODER_B_R = 19, 26

    # Inversões (chassi espelhado)
    INVERT_LEFT  = False
    INVERT_RIGHT = True  # direita invertida

    def __init__(self, config):
        self.config = config or {}
        # Força backend lgpio (imune ao bloqueio do rpi_ws281x)
        self.GPIO, self.backend = _make_backend_forced_lgpio()

        # estado
        self._ticks_l = 0; self._ticks_r = 0
        self.meas_tps_l = 0.0; self.meas_tps_r = 0.0
        self.target_tps_l = 0.0; self.target_tps_r = 0.0
        self.last_left_speed = 0; self.last_right_speed = 0
        self._encoders_ok = True

        self._pwm_left = None; self._pwm_right = None
        self._servo_pwm = [None]*4; self._servo_ready=[False]*4

        self._safe_setup_io()

        self.dir_pid = PIDController(
            kp=self.config.get("pid", {}).get("kp", 0.9),
            ki=self.config.get("pid", {}).get("ki", 0.0),
            kd=self.config.get("pid", {}).get("kd", 0.14),
            sample_time=self.config.get("pid", {}).get("sample_time", 0.02)
        )
        self.pid_vel_l = PIDController(out_min=-100.0, out_max=100.0, **self.VEL_PID_L)
        self.pid_vel_r = PIDController(out_min=-100.0, out_max=100.0, **self.VEL_PID_R)

        self._init_servos()

        self._last_meas_t = time.time()
        self._last_ticks_l = 0; self._last_ticks_r = 0
        self._enc_dead_timer = None

        self._ctrl_running = True
        self._ctrl_th = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_th.start()

    # ---------- backend helpers ----------
    def _safe_setup_io(self):
        # motores
        for pin in [self.L_BIN1,self.L_BIN2,self.R_BIN1,self.R_BIN2,self.STBY]:
            self.GPIO.setup(pin, self.GPIO.OUT)
        self.GPIO.setup(self.L_PWMB, self.GPIO.OUT)
        self.GPIO.setup(self.R_PWMB, self.GPIO.OUT)

        # PWM 500 Hz
        self._pwm_left = self.GPIO.PWM(self.L_PWMB, 500)
        self._pwm_right = self.GPIO.PWM(self.R_PWMB, 500)
        self._pwm_left.start(0); self._pwm_right.start(0)

        # STBY alto (no seu hardware já fica ligado; garantimos HIGH)
        self.GPIO.output(self.STBY, 1)

        # encoders (opcionais)
        try:
            self.GPIO.setup(self.ENCODER_A_L, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP)
            self.GPIO.setup(self.ENCODER_A_R, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP)
            self.GPIO.add_event_detect(self.ENCODER_A_L, self.GPIO.RISING, callback=self._enc_l)
            self.GPIO.add_event_detect(self.ENCODER_A_R, self.GPIO.RISING, callback=self._enc_r)
            self._encoders_ok = True
        except Exception:
            self._encoders_ok = False

    # ---------- encoders ----------
    def _enc_l(self, _): self._ticks_l += 1
    def _enc_r(self, _): self._ticks_r += 1
    def reset_encoders(self):
        self._ticks_l = 0
        self._ticks_r = 0
        self._last_ticks_l = 0
        self._last_ticks_r = 0
        self.meas_tps_l = 0.0
        self.meas_tps_r = 0.0
    def read_encoders(self): return self._ticks_l, self._ticks_r
    def read_speeds_tps(self): return self.meas_tps_l, self.meas_tps_r

    # ---------- util ----------
    def _apply_deadband(self, cmd: float) -> float:
        cmd = float(cmd)
        if cmd == 0.0:
            return 0.0
        a = abs(cmd)
        if a < self.MIN_DUTY:
            return self.MIN_DUTY * (1.0 if cmd > 0 else -1.0)
        return cmd

    def set_open_loop(self, enabled: bool):
        """Permite alternar entre open-loop (PWM direto) e closed-loop (encoders)."""
        self.OPEN_LOOP = bool(enabled)

    # ---------- motores ----------
    def set_motor_speed(self, base_speed, error):
        # mistura base/erro via PID direcional
        correction = self.dir_pid.calculate(error)
        left = float(base_speed) - float(correction)
        right = float(base_speed) + float(correction)

        # limita
        left = max(-100.0, min(100.0, left))
        right = max(-100.0, min(100.0, right))

        # snap-to-straight (erro pequeno e comandos parecidos) → 80/80
        if abs(error) < self.STRAIGHT_ERR_TH and abs(left - right) < self.STRAIGHT_MATCH_DELTA:
            left = self.STRAIGHT_MIN_DUTY
            right = self.STRAIGHT_MIN_DUTY

        # deadband (garante saída mesmo com comando baixo)
        left = self._apply_deadband(left) if left != 0 else 0.0
        right = self._apply_deadband(right) if right != 0 else 0.0

        # telemetria p/ UI
        self.last_left_speed = int(left)
        self.last_right_speed = int(right)

        # open-loop (padrão) ou closed-loop se encoders OK e ativado
        if self.OPEN_LOOP or not self._encoders_ok:
            self._write_motor_pwm(left, right)
        else:
            self.target_tps_l = (left / 100.0) * self.MAX_TICKS_PER_SEC
            self.target_tps_r = (right / 100.0) * self.MAX_TICKS_PER_SEC
            # watchdog: se alvo alto e tps ~0 por FALLBACK_CHECK_SEC -> open loop
            if self.AUTO_FALLBACK_OPEN and (abs(self.target_tps_l) > 50 or abs(self.target_tps_r) > 50):
                if (self.meas_tps_l < 5 and self.meas_tps_r < 5):
                    if self._enc_dead_timer is None:
                        self._enc_dead_timer = time.time()
                    elif time.time() - self._enc_dead_timer >= self.FALLBACK_CHECK_SEC:
                        self.OPEN_LOOP = True
                else:
                    self._enc_dead_timer = None

    def stop(self):
        self.target_tps_l = 0.0
        self.target_tps_r = 0.0
        self._write_motor_pwm(0.0, 0.0)

    def _control_loop(self):
        period = 1.0 / float(self.CONTROL_HZ)
        pwm_l = 0.0; pwm_r = 0.0
        alpha = 0.4
        while self._ctrl_running:
            t0 = time.time()
            if not self.OPEN_LOOP and self._encoders_ok:
                now = time.time()
                dt = now - self._last_meas_t
                if dt <= 0: dt = period

                ticks_l = self._ticks_l; ticks_r = self._ticks_r
                dt_l = ticks_l - getattr(self, "_last_ticks_l", 0)
                dt_r = ticks_r - getattr(self, "_last_ticks_r", 0)

                inst_l = float(dt_l) / dt
                inst_r = float(dt_r) / dt

                self.meas_tps_l = alpha * inst_l + (1 - alpha) * self.meas_tps_l
                self.meas_tps_r = alpha * inst_r + (1 - alpha) * self.meas_tps_r

                self._last_ticks_l = ticks_l; self._last_ticks_r = ticks_r
                self._last_meas_t = now

                self.pid_vel_l.setpoint = self.target_tps_l
                self.pid_vel_r.setpoint = self.target_tps_r

                adj_l = self.pid_vel_l.calculate(self.meas_tps_l)
                adj_r = self.pid_vel_r.calculate(self.meas_tps_r)

                pwm_l = max(-100.0, min(100.0, pwm_l + adj_l * 0.2))
                pwm_r = max(-100.0, min(100.0, pwm_r + adj_r * 0.2))

                pwm_l = self._apply_deadband(pwm_l) if pwm_l != 0 else 0.0
                pwm_r = self._apply_deadband(pwm_r) if pwm_r != 0 else 0.0

                self._write_motor_pwm(pwm_l, pwm_r)

            # dorme até completar o período
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _write_motor_pwm(self, left_cmd, right_cmd):
        """
        Aplica inversão por roda antes de dirigir BIN1/BIN2.
        CMD >= 0 -> frente (BIN1=1, BIN2=0)
        CMD <  0 -> ré     (BIN1=0, BIN2=1)
        """
        # inversões (chassi espelhado)
        if self.INVERT_LEFT:
            left_cmd = -left_cmd
        if self.INVERT_RIGHT:
            right_cmd = -right_cmd

        # ESQUERDA
        if left_cmd >= 0:
            self.GPIO.output(self.L_BIN1, 1)
            self.GPIO.output(self.L_BIN2, 0)
        else:
            self.GPIO.output(self.L_BIN1, 0)
            self.GPIO.output(self.L_BIN2, 1)
        self._pwm_left.ChangeDutyCycle(abs(left_cmd))

        # DIREITA
        if right_cmd >= 0:
            self.GPIO.output(self.R_BIN1, 1)
            self.GPIO.output(self.R_BIN2, 0)
        else:
            self.GPIO.output(self.R_BIN1, 0)
            self.GPIO.output(self.R_BIN2, 1)
        self._pwm_right.ChangeDutyCycle(abs(right_cmd))

    # ---------- servos ----------
    @staticmethod
    def _us_to_duty(us, period_ms=20.0):
        return max(0.0, min(100.0, (us / 1000.0) / period_ms * 100.0))

    def _servo_write_us(self, index, us):
        i = index - 1
        if not (0 <= i < 4) or not self._servo_ready[i]:
            return False
        self._servo_pwm[i].ChangeDutyCycle(self._us_to_duty(int(us)))
        return True

    def set_servo(self, index, pos, smooth_ms=0):
        i = index - 1
        if not (0 <= i < 4) or pos not in ("A","B"):
            return False
        target = int(self.SERVO_AB_US[i][pos])
        if smooth_ms > 0:
            other = 'A' if pos == 'B' else 'B'
            start = int(self.SERVO_AB_US[i][other])
            steps = max(3, int(smooth_ms / 20))
            for t in range(steps + 1):
                u = int(start + (target - start) * (t / steps))
                self._servo_write_us(index, u)
                time.sleep(0.02)
        else:
            self._servo_write_us(index, target)
        return True

    def set_servos(self, positions=('A','A','A','A'), smooth_ms=0):
        for idx, p in enumerate(positions, start=1):
            self.set_servo(idx, p, smooth_ms=smooth_ms)

    def set_servo_us(self, index, us):
        return self._servo_write_us(index, int(us))

    def _init_servos(self):
        """Inicializa PWM dos 4 servos e coloca todos em 'A'. Se falhar, marca como indisponível."""
        for i, pin in enumerate(self.SERVO_PINS):
            try:
                self.GPIO.setup(pin, self.GPIO.OUT)
                pwm = self.GPIO.PWM(pin, self.SERVO_FREQ_HZ)
                pwm.start(0)
                self._servo_pwm[i] = pwm
                self._servo_ready[i] = True
                try:
                    self.set_servo(i + 1, 'A', smooth_ms=150)
                except Exception:
                    pass
            except Exception:
                self._servo_ready[i] = False
        time.sleep(0.05)

    # ---------- cleanup ----------
    def cleanup(self):
        self._ctrl_running = False
        try:
            if hasattr(self,"_ctrl_th") and self._ctrl_th and self._ctrl_th.is_alive():
                self._ctrl_th.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.stop()
        except Exception:
            pass
        try:
            for pwm in self._servo_pwm:
                if pwm: pwm.stop()
        except Exception:
            pass
        try:
            if self._pwm_left: self._pwm_left.stop()
            if self._pwm_right: self._pwm_right.stop()
        except Exception:
            pass
        try:
            self.GPIO.cleanup()
        except Exception:
            pass
