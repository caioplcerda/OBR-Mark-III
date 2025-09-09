import RPi.GPIO as GPIO
import time

# GPIO pin where your servo is connected
SERVO_PIN = 20  # change if needed

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# 50 Hz PWM for servo
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_angle(angle):
    # Convert angle (0-180) to duty cycle (2-12)
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)

try:
    print("Moving to 0°")
    set_angle(0)
    
    print("Moving to 90°")
    set_angle(90)
    
    print("Moving to 180°")
    set_angle(180)
    
    print("Test complete. Observe if the servo stops at angles or keeps spinning.")
    
finally:
    pwm.stop()
    GPIO.cleanup()
