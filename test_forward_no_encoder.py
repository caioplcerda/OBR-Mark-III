import RPi.GPIO as GPIO
import time

# Pin definitions for two TB6612FNG motor drivers (both using channel B)
# Left motor driver
L_BIN1 = 17
L_BIN2 = 27
L_PWMB = 22

# Right motor driver
R_BIN1 = 23
R_BIN2 = 24
R_PWMB = 25

# Shared standby pin
STBY = 5


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup motor pins
    for pin in [L_BIN1, L_BIN2, R_BIN1, R_BIN2, STBY]:
        GPIO.setup(pin, GPIO.OUT)
    GPIO.setup(L_PWMB, GPIO.OUT)
    GPIO.setup(R_PWMB, GPIO.OUT)

    pwm_left = GPIO.PWM(L_PWMB, 100)
    pwm_right = GPIO.PWM(R_PWMB, 100)
    pwm_left.start(0)
    pwm_right.start(0)

    # Enable motor driver
    GPIO.output(STBY, GPIO.HIGH)

    try:
        # Drive both motors forward at 50% duty cycle
        GPIO.output(L_BIN1, GPIO.LOW)
        GPIO.output(L_BIN2, GPIO.HIGH)
        GPIO.output(R_BIN1, GPIO.HIGH)
        GPIO.output(R_BIN2, GPIO.LOW)
        pwm_left.ChangeDutyCycle(100)
        pwm_right.ChangeDutyCycle(100)

        print("Motors running forward. Press Ctrl+C to stop.")
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if pwm_left:
            pwm_left.stop()
            pwm_left = None
        if pwm_right:
            pwm_right.stop()
            pwm_right = None
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
