import RPi.GPIO as GPIO
import time
import os

class PIDController:
    """ Controlador PID para o robô. """
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
        """ Calcula a correção PID com base no valor atual. """
        now = time.time()
        dt = now - self.last_time

        if dt < self.sample_time:
            return self.prev_error

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # Anti-windup para evitar que o termo integral cresça demais
        if self.integral > 100: self.integral = 100
        if self.integral < -100: self.integral = -100

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        self.last_time = now

        return output

class HardwareControl:
    """ Classe para controlar todo o hardware do robô. """
    def __init__(self):
        # === Pinos GPIO para os motores ===
        self.LEFT_MOTOR_FORWARD = 17
        self.LEFT_MOTOR_BACKWARD = 27
        self.RIGHT_MOTOR_FORWARD = 22
        self.RIGHT_MOTOR_BACKWARD = 23
        self.FAN_GPIO = 24

        # === Pinos GPIO para os servos (a serem definidos) ===
        self.SERVO_GARRA_1 = None
        self.SERVO_GARRA_2 = None
        self.SERVO_RESERVATORIO = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LEFT_MOTOR_FORWARD, GPIO.OUT)
        GPIO.setup(self.LEFT_MOTOR_BACKWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_MOTOR_FORWARD, GPIO.OUT)
        GPIO.setup(self.RIGHT_MOTOR_BACKWARD, GPIO.OUT)
        GPIO.setup(self.FAN_GPIO, GPIO.OUT)
        GPIO.output(self.FAN_GPIO, GPIO.HIGH) # Liga a ventoinha

        # === PWM para os motores ===
        self.left_pwm_fwd = GPIO.PWM(self.LEFT_MOTOR_FORWARD, 100)
        self.right_pwm_fwd = GPIO.PWM(self.RIGHT_MOTOR_FORWARD, 100)
        self.left_pwm_fwd.start(0)
        self.right_pwm_fwd.start(0)

        # === PWM para os servos ===
        self.servos = []
        for pin in [self.SERVO_GARRA_1, self.SERVO_GARRA_2, self.SERVO_RESERVATORIO]:
            if pin is not None:
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, 50)
                pwm.start(0)
                self.servos.append(pwm)
            else:
                self.servos.append(None)

        self.activate_pi_fan()
        self.pid_controller = PIDController()

    def activate_pi_fan(self):
        """ Ativa a ventoinha do Raspberry Pi 5. """
        try:
            if os.path.exists("/usr/bin/rpi-fancontrol"):
                os.system("sudo rpi-fancontrol --fan 1")
            elif os.path.exists("/proc/device-tree/thermal-zones/fan-thermal/cooling-device"):
                os.system("echo 1 | sudo tee /sys/class/thermal/cooling_device0/cur_state")
            else:
                print("[INFO] Ventoinha embutida não detectada.")
        except Exception as e:
            print(f"[WARNING] Erro ao ativar ventoinha: {e}")

    def set_motor_speed(self, base_speed, error):
        """ Define a velocidade dos motores com base na velocidade base e no erro PID. """
        correction = self.pid_controller.calculate(error)

        left_speed = base_speed + correction
        right_speed = base_speed - correction

        left_speed = max(0, min(100, left_speed))
        right_speed = max(0, min(100, right_speed))

        self.left_pwm_fwd.ChangeDutyCycle(left_speed)
        self.right_pwm_fwd.ChangeDutyCycle(right_speed)

    def stop(self):
        """ Para os motores. """
        self.left_pwm_fwd.ChangeDutyCycle(0)
        self.right_pwm_fwd.ChangeDutyCycle(0)

    def set_servo_angle(self, servo_index, angle):
        """ Define o ângulo de um servo motor. """
        pwm = self.servos[servo_index]
        if pwm:
            duty = 2 + (angle / 18)
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.5)
            pwm.ChangeDutyCycle(0)

    def cleanup(self):
        """ Limpa os pinos GPIO ao encerrar. """
        self.stop()
        GPIO.cleanup()
