import RPi.GPIO as GPIO
import time

# Pin definitions for TB6612FNG motor driver
AIN1 = 17
AIN2 = 27
PWMA = 22
BIN1 = 23
BIN2 = 24
PWMB = 25
STBY = 5


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup motor pins
    for pin in [AIN1, AIN2, BIN1, BIN2, STBY]:
        GPIO.setup(pin, GPIO.OUT)
    GPIO.setup(PWMA, GPIO.OUT)
    GPIO.setup(PWMB, GPIO.OUT)

    pwm_a = GPIO.PWM(PWMA, 100)
    pwm_b = GPIO.PWM(PWMB, 100)
    pwm_a.start(0)
    pwm_b.start(0)

    # Enable motor driver
    GPIO.output(STBY, GPIO.HIGH)

    try:
        # Drive both motors forward at 50% duty cycle
        GPIO.output(AIN1, GPIO.HIGH)
        GPIO.output(AIN2, GPIO.LOW)
        GPIO.output(BIN1, GPIO.HIGH)
        GPIO.output(BIN2, GPIO.LOW)
        pwm_a.ChangeDutyCycle(50)
        pwm_b.ChangeDutyCycle(50)
        print("Motors running forward. Press Ctrl+C to stop.")
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if pwm_a:
            pwm_a.stop()
            pwm_a = None
        if pwm_b:
            pwm_b.stop()
            pwm_b = None
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
