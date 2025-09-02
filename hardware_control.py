import RPi.GPIO as GPIO
import time
import threading
import os

# =========================
#   PID (posição/erro e velocidade)
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
            # retorna último erro como “output” para não travar quem chamou
            return self._prev_error

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
    - TB6612FNG: pinos iguais aos seus
    - Encoders em loop fechado de velocidade (ticks/seg)
    - Mantém API: set_motor_speed(base_speed, error)
    - 4 servos (duas posições) — exatamente como antes
    """

    # ======= CONFIG DE MOTOR/ENCODER =======
    # Ajuste estes 3 conforme o seu conjunto motor/encoder:
    TICKS_PER_REV = 20           # nº de pulsos por volta (canal A). Troque p/ o valor do seu encoder!
    MAX_TICKS_PER_SEC = 300.0    # ticks/s que você atinge a ~100% PWM em linha reta
    CONTROL_HZ = 50              # frequência do laço de velocidade (20ms)

    # PID de velocidade por roda (saída = ajuste de PWM)
    VEL_PID_L = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)
    VEL_PID_R = dict(kp=0.25, ki=0.35, kd=0.0, sample_time=1.0/CONTROL_HZ)

    # ======= SERVOS =======
    SERVO_PINS = [7, 18, 16, 20]   # 7 no lugar do 12 (12 está nos LEDs WS2812)
    SERVO_FREQ_HZ = 50
    SERVO_AB_US = [
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
        {"A": 1000, "B": 2000},
    ]

    def __init__(self, config):
        self.config = config or {}

        # === Pinos TB6612FNG (canal B de cada placa) ===
        # Esquerdo
        self.L_BIN1 = 17
        self.L_BIN2 = 27
        self.L_PWMB = 22
        # Direito
        self.R_BIN1 = 23
        self.R_BIN2 = 24
        self.R_PWMB = 25
        # Standby
        self.STBY = 5

        # Botão (não usado pelo main)
        self.START_BUTTON = 4

        # Encoders (canal A com interrupção)
        self.ENCODER_A_L = 6
        self.ENCODER_A_R = 19

        # Estado dos encoders
        self._ticks_l = 0
        self._ticks_r = 0

        # Medidas filtradas (ticks/s)
        self.meas_tps_l = 0.0
        self.meas_tps_r = 0.0

        # Alvos de velocidade (ticks/s)
        self.target_tps_l = 0.0
        self.target_tps_r = 0.0

        # Últimos comandos “conforto/telemetria”
        self.last_left_speed = 0   # duty/sinal final [-100..100]
        self.last_right_speed = 0

        # GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        # Motores
        for pin in [self.L_BIN1, self.L_BIN2, self.R_BIN1, self.R_BIN2, self.STBY]:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.L_PWMB, GPIO.OUT)
        GPIO.setup(self.R_PWMB, GPIO.OUT)

        # Botão (opcional, não usado pelo main)
        GPIO.setup(self.START_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Encoders (canal A com contagem por borda de subida)
        GPIO.setup(self.ENCODER_A_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.ENCODER_A_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.ENCODER_A_L, GPIO.RISING, callback=self._enc_l)
        GPIO.add_event_detect(self.ENCODER_A_R, GPIO.RISING, callback=self._enc_r)

        # PWM motores
        self._pwm_left = GPIO.PWM(self.L_PWMB, 1000)   # 1 kHz dá resposta suave
        self._pwm_right = GPIO.PWM(self.R_PWMB, 1000)
        self._pwm_left.start(0)
        self._pwm_right.start(0)

        # Ativa o driver
        GPIO.output(self.STBY, GPIO.HIGH)

        # PID “direcional” (base+erro) vindo do main → mantemos para mistura
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

        # Thread de controle de velocidade
        self._ctrl_running = True
        self._ctrl_th = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_th.start()

        # Auxiliar para medição de TPS
        self._last_meas_t = time.time()
        self._last_ticks_l = 0
        self._last_ticks_r = 0

    # --------------------- Encoders callbacks ---------------------
    def _enc_l(self, ch):
        self._ticks_l += 1

    def _enc_r(self, ch):
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
        """Velocidade medida (ticks/s) de cada roda (filtrada ~EMA)."""
        return self.meas_tps_l, self.meas_tps_r

    # --------------------- Motores (API esperada pelo main) -------
    def set_motor_speed(self, base_speed, error):
        """
        Recebe (base, erro) do controlador de linha.
        Converte em comandos por roda e define ALVOS de velocidade (ticks/s).
        O laço interno (_control_loop) usa os encoders para acertar o PWM.
        """
        # Saída “direcional” do PID do main (opcional, mas mantido):
        correction = self.dir_pid.calculate(error)

        # Comandos “virtuais” por roda (–100..100):
        cmd_left = float(base_speed) - float(correction)
        cmd_right = float(base_speed) + float(correction)

        # Limita:
        cmd_left = max(-100.0, min(100.0, cmd_left))
        cmd_right = max(-100.0, min(100.0, cmd_right))

        # Guarda para painel
        self.last_left_speed = int(cmd_left)
        self.last_right_speed = int(cmd_right)

        # Converte em velocidade‐alvo por roda (ticks/s)
        self.target_tps_l = (cmd_left / 100.0) * self.MAX_TICKS_PER_SEC
        self.target_tps_r = (cmd_right / 100.0) * self.MAX_TICKS_PER_SEC

    def stop(self):
        # desliga alvos e PWM
        self.target_tps_l = 0.0
        self.target_tps_r = 0.0
        self._write_motor_pwm(0.0, 0.0, forward_hint=True)

    # --------------------- Laço de velocidade por roda ------------
    def _control_loop(self):
        """
        Roda em ~CONTROL_HZ:
        - calcula velocidade medida (ticks/s) por delta de pulsos
        - PIDs de velocidade ajustam o PWM por roda
        """
        period = 1.0 / float(self.CONTROL_HZ)
        # Estado interno de PWM (–100..100)
        pwm_l = 0.0
        pwm_r = 0.0

        # Filtro EMA para medida (suaviza ruído)
        alpha = 0.4

        while self._ctrl_running:
            t0 = time.time()

            # Mede ticks/s
            now = time.time()
            dt = now - self._last_meas_t
            if dt <= 0: dt = period

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

            # PIDs de velocidade: setpoints são target_tps_*
            self.pid_vel_l.setpoint = self.target_tps_l
            self.pid_vel_r.setpoint = self.target_tps_r

            adj_l = self.pid_vel_l.calculate(self.meas_tps_l)   # –100..100 (limitado no PID)
            adj_r = self.pid_vel_r.calculate(self.meas_tps_r)

            # Atualiza PWM aplicando ajuste (aceleração suave)
            pwm_l = max(-100.0, min(100.0, pwm_l + adj_l * 0.2))
            pwm_r = max(-100.0, min(100.0, pwm_r + adj_r * 0.2))

            # Escreve nos motores
            self._write_motor_pwm(pwm_l, pwm_r)

            # Dorme até completar o período
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _write_motor_pwm(self, left_cmd, right_cmd, forward_hint=False):
        """
        left_cmd/right_cmd ∈ [-100..100] → define sentido (BIN1/BIN2) e duty.
        Observação: seu lado direito é invertido por fio — mantive essa convenção.
        """
        # Esquerda
        if left_cmd >= 0:
            GPIO.output(self.L_BIN1, GPIO.HIGH)
            GPIO.output(self.L_BIN2, GPIO.LOW)
        else:
            GPIO.output(self.L_BIN1, GPIO.LOW)
            GPIO.output(self.L_BIN2, GPIO.HIGH)
        self._pwm_left.ChangeDutyCycle(abs(left_cmd))

        # Direita (invertido)
        if right_cmd >= 0:
            GPIO.output(self.R_BIN1, GPIO.LOW)
            GPIO.output(self.R_BIN2, GPIO.HIGH)
        else:
            GPIO.output(self.R_BIN1, GPIO.HIGH)
            GPIO.output(self.R_BIN2, GPIO.LOW)
        self._pwm_right.ChangeDutyCycle(abs(right_cmd))

        # Standby sempre ativo quando há comando
        if abs(left_cmd) > 0.1 or abs(right_cmd) > 0.1:
            GPIO.output(self.STBY, GPIO.HIGH)
        else:
            GPIO.output(self.STBY, GPIO.LOW)

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
        self._servo_pwm = [None, None, None, None]
        self._servo_ready = [False, False, False, False]
        for i, pin in enumerate(self.SERVO_PINS):
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, self.SERVO_FREQ_HZ)
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
            if self._ctrl_th and self._ctrl_th.is_alive():
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

        GPIO.cleanup()
