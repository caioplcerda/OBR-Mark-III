import RPi.GPIO as GPIO
import time
import os

# =========================
#   PID (mesmo do seu)
# =========================
class PIDController:
    def __init__(self, kp=0.4, ki=0.0, kd=0.1, setpoint=0, sample_time=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def calculate(self, current_value):
        now = time.time()
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.prev_error

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # anti-windup simples
        self.integral = max(-100, min(100, self.integral))

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        self.last_time = now
        return output


# =========================
#   HardwareControl
# =========================
class HardwareControl:
    """
    Motores (TB6612FNG), Encoders, Botão start opcional (se quiser usar no main),
    e **4 SERVOS** com duas posições fixas (A/B) por servo.
    """

    # ---------- CONFIG SERVOS ----------
    # Pinos S1..S4 (mude aqui se quiser)
    SERVO_PINS = [12, 18, 16, 20]   # BCM (12/18 = HW PWM; 16/20 ok via PWM de software do RPi.GPIO)
    SERVO_FREQ_HZ = 50              # 50Hz = período ~20ms

    # Posições padrão em microssegundos (ajuste por servo se necessário)
    # A = posição 1 (ex.: retraído/fechado), B = posição 2 (ex.: estendido/aberto)
    SERVO_AB_US = [
        {"A": 1000, "B": 2000},  # S1
        {"A": 1000, "B": 2000},  # S2
        {"A": 1000, "B": 2000},  # S3
        {"A": 1000, "B": 2000},  # S4
    ]

    def __init__(self, config):
        self.config = config

        # === Pinos para os Drivers TB6612FNG (canal B de cada placa) ===
        # Driver do motor esquerdo
        self.L_BIN1 = 17
        self.L_BIN2 = 27
        self.L_PWMB = 22
        # Driver do motor direito
        self.R_BIN1 = 23
        self.R_BIN2 = 24
        self.R_PWMB = 25
        # Pino de standby compartilhado
        self.STBY = 5

        # (Opcional) Botão de início deste módulo (não usado pelo main)
        self.START_BUTTON = 4

        # === Pinos para os Encoders ===
        self.ENCODER_A_L = 6
        self.ENCODER_B_L = 13
        self.ENCODER_A_R = 19
        self.ENCODER_B_R = 26

        # === Contadores de Pulso dos Encoders ===
        self.encoder_ticks_l = 0
        self.encoder_ticks_r = 0

        # === Últimas velocidades definidas ===
        self.last_left_speed = 0
        self.last_right_speed = 0

        # GPIO base
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        # Motores
        for pin in [self.L_BIN1, self.L_BIN2, self.R_BIN1, self.R_BIN2, self.STBY]:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.L_PWMB, GPIO.OUT)
        GPIO.setup(self.R_PWMB, GPIO.OUT)

        # Botão (se quiser usar aqui)
        GPIO.setup(self.START_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Encoders (canal A com detecção; B se quiser direção, já está declarado)
        GPIO.setup(self.ENCODER_A_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.ENCODER_A_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.ENCODER_A_L, GPIO.RISING, callback=self.encoder_callback_l)
        GPIO.add_event_detect(self.ENCODER_A_R, GPIO.RISING, callback=self.encoder_callback_r)

        # PWM motores
        self.pwm_left = GPIO.PWM(self.L_PWMB, 100)
        self.pwm_right = GPIO.PWM(self.R_PWMB, 100)
        self.pwm_left.start(0)
        self.pwm_right.start(0)

        # Ativa o driver
        GPIO.output(self.STBY, GPIO.HIGH)

        # PID
        self.pid_controller = PIDController()
        self.update_pid_from_config()

        # ====== SERVOS ======
        self._servo_pwm = [None, None, None, None]  # objetos PWM
        self._servo_ready = [False, False, False, False]
        self._init_servos()

    # --------------------- PID config ---------------------
    def update_pid_from_config(self):
        pid_params = self.config.get('pid', {})
        self.pid_controller.kp = pid_params.get('kp', self.pid_controller.kp)
        self.pid_controller.ki = pid_params.get('ki', self.pid_controller.ki)
        self.pid_controller.kd = pid_params.get('kd', self.pid_controller.kd)

    # --------------------- Encoders -----------------------
    def encoder_callback_l(self, channel):
        self.encoder_ticks_l += 1

    def encoder_callback_r(self, channel):
        self.encoder_ticks_r += 1

    # --------------------- Motores ------------------------
    def set_motor_speed(self, base_speed, error):
        """Interface esperada pelo main: (base, erro) -> PID -> left/right."""
        correction = self.pid_controller.calculate(error)

        left_speed = base_speed - correction
        right_speed = base_speed + correction

        # salva pra UI
        self.last_left_speed = int(max(-100, min(100, left_speed)))
        self.last_right_speed = int(max(-100, min(100, right_speed)))

        # Esquerdo
        if self.last_left_speed >= 0:
            GPIO.output(self.L_BIN1, GPIO.HIGH)
            GPIO.output(self.L_BIN2, GPIO.LOW)
        else:
            GPIO.output(self.L_BIN1, GPIO.LOW)
            GPIO.output(self.L_BIN2, GPIO.HIGH)
        self.pwm_left.ChangeDutyCycle(abs(self.last_left_speed))

        # Direito (invertido)
        if self.last_right_speed >= 0:
            GPIO.output(self.R_BIN1, GPIO.LOW)
            GPIO.output(self.R_BIN2, GPIO.HIGH)
        else:
            GPIO.output(self.R_BIN1, GPIO.HIGH)
            GPIO.output(self.R_BIN2, GPIO.LOW)
        self.pwm_right.ChangeDutyCycle(abs(self.last_right_speed))

    def stop(self):
        GPIO.output(self.STBY, GPIO.LOW)
        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)

    # --------------------- Servos -------------------------
    def _init_servos(self):
        """Inicializa os 4 servos nos pinos definidos com 50Hz."""
        for i, pin in enumerate(self.SERVO_PINS):
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, self.SERVO_FREQ_HZ)
            pwm.start(0)
            self._servo_pwm[i] = pwm
            self._servo_ready[i] = True
            # coloca em posição A por padrão (suave)
            try:
                self.set_servo(i+1, 'A', smooth_ms=200)
            except Exception:
                pass
        # pequeno descanso para estabilizar
        time.sleep(0.15)

    @staticmethod
    def _us_to_duty(us, period_ms=20.0):
        """Converte microssegundos para duty-cycle (%) em 50Hz (~20ms)."""
        return max(0.0, min(100.0, (us / 1000.0) / period_ms * 100.0))

    def _servo_write_us(self, index, us):
        """Escreve pulso em microssegundos no servo index (1..4)."""
        i = index - 1
        if not (0 <= i < 4) or not self._servo_ready[i]:
            return False
        duty = self._us_to_duty(us)
        self._servo_pwm[i].ChangeDutyCycle(duty)
        return True

    def set_servo(self, index, pos, smooth_ms=0):
        """
        Define servo index (1..4) para posição 'A' ou 'B'.
        smooth_ms: se >0, rampa suave ao longo desse tempo.
        """
        i = index - 1
        if not (0 <= i < 4):
            return False
        if pos not in ('A', 'B'):
            return False

        target_us = int(self.SERVO_AB_US[i][pos])

        if smooth_ms and smooth_ms > 0:
            # leitura aproximada do atual (não temos feedback -> estimamos a partir do último comando)
            # estratégia: gerar pequenos passos do valor atual para o alvo
            # para evitar "pulos", calculamos a partir da outra posição como “ponto de partida” plausível
            other = 'A' if pos == 'B' else 'B'
            start_us = int(self.SERVO_AB_US[i][other])
            steps = max(3, int(smooth_ms / 20))  # ~20ms por passo
            for t in range(steps+1):
                u = int(start_us + (target_us - start_us) * (t / steps))
                self._servo_write_us(index, u)
                time.sleep(0.02)
        else:
            self._servo_write_us(index, target_us)

        return True

    def set_servos(self, positions=('A', 'A', 'A', 'A'), smooth_ms=0):
        """Define todos os 4 servos de uma vez. positions: tupla/lista com 'A'/'B'."""
        for idx, p in enumerate(positions, start=1):
            self.set_servo(idx, p, smooth_ms=smooth_ms)

    def set_servo_us(self, index, us):
        """Ajuste fino em microssegundos (útil para calibração)."""
        return self._servo_write_us(index, int(us))

    def calibrate_servo(self, index, A_us=None, B_us=None, commit=True):
        """Atualiza os limites A/B (us) do servo index. commit=True salva na memória do objeto."""
        i = index - 1
        if not (0 <= i < 4):
            return False
        if A_us is not None:
            self.SERVO_AB_US[i]['A'] = int(A_us)
        if B_us is not None:
            self.SERVO_AB_US[i]['B'] = int(B_us)
        return True

    def release_servos(self):
        """Para o PWM dos servos (servos soltos – podem se mover)."""
        for i, pwm in enumerate(self._servo_pwm):
            if pwm:
                pwm.ChangeDutyCycle(0)

    # --------------------- Cleanup ------------------------
    def cleanup(self):
        self.stop()
        self.release_servos()
        # encerra PWM de servos
        for pwm in self._servo_pwm:
            try:
                if pwm:
                    pwm.stop()
            except Exception:
                pass
        # encerra PWM de motores
        try:
            self.pwm_left.stop()
            self.pwm_right.stop()
        except Exception:
            pass
        GPIO.cleanup()
