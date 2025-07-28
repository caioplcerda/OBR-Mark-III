import RPi.GPIO as GPIO
import time
import os

class PIDController:
    # ... (código do PIDController permanece o mesmo) ...
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

        if self.integral > 100: self.integral = 100
        if self.integral < -100: self.integral = -100

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        self.last_time = now

        return output

class HardwareControl:
    def __init__(self, config):
        self.config = config
        # === Pinos para o Driver TB6612FNG ===
        self.AIN1 = 17
        self.AIN2 = 27
        self.PWMA = 22
        self.BIN1 = 23
        self.BIN2 = 24
        self.PWMB = 25
        self.STBY = 5

        # === Pinos para os Encoders ===
        self.ENCODER_A_L = 6
        self.ENCODER_B_L = 13
        self.ENCODER_A_R = 19
        self.ENCODER_B_R = 26

        # === Contadores de Pulso dos Encoders ===
        self.encoder_ticks_l = 0
        self.encoder_ticks_r = 0

        GPIO.setmode(GPIO.BCM)
        # Configura pinos do motor
        for pin in [self.AIN1, self.AIN2, self.BIN1, self.BIN2, self.STBY]:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.PWMA, GPIO.OUT)
        GPIO.setup(self.PWMB, GPIO.OUT)

        # Configura pinos do encoder
        GPIO.setup(self.ENCODER_A_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.ENCODER_A_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Adiciona interrupções para os encoders
        GPIO.add_event_detect(self.ENCODER_A_L, GPIO.RISING, callback=self.encoder_callback_l)
        GPIO.add_event_detect(self.ENCODER_A_R, GPIO.RISING, callback=self.encoder_callback_r)

        # PWM para os motores
        self.pwm_a = GPIO.PWM(self.PWMA, 100)
        self.pwm_b = GPIO.PWM(self.PWMB, 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

        # Ativa o driver
        GPIO.output(self.STBY, GPIO.HIGH)

        self.pid_controller = PIDController()
        self.update_pid_from_config()

    def update_pid_from_config(self):
        pid_params = self.config['pid']
        self.pid_controller.kp = pid_params['kp']
        self.pid_controller.ki = pid_params['ki']
        self.pid_controller.kd = pid_params['kd']

    def encoder_callback_l(self, channel):
        self.encoder_ticks_l += 1

    def encoder_callback_r(self, channel):
        self.encoder_ticks_r += 1

    def set_motor_speed(self, base_speed, error):
        correction = self.pid_controller.calculate(error)

        left_speed = base_speed - correction
        right_speed = base_speed + correction

        # Controle do Motor A (Esquerdo)
        if left_speed > 0:
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
        else:
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(min(abs(left_speed), 100))

        # Controle do Motor B (Direito)
        if right_speed > 0:
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
        else:
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)
        self.pwm_b.ChangeDutyCycle(min(abs(right_speed), 100))

    def stop(self):
        GPIO.output(self.STBY, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()

    # As funções de servo e ventoinha foram removidas para simplificar,
    # mas podem ser adicionadas novamente se necessário.
