import pigpio
import time

pi = pigpio.pi()
SERVO_PIN = 18  # GPIO pin

# 1000 µs = 0°, 1500 µs = 90°, 2000 µs = 180°
pi.set_servo_pulsewidth(SERVO_PIN, 1000)
time.sleep(1)
pi.set_servo_pulsewidth(SERVO_PIN, 1500)
time.sleep(1)
pi.set_servo_pulsewidth(SERVO_PIN, 2000)
time.sleep(1)

pi.set_servo_pulsewidth(SERVO_PIN, 0)  # stop signal
pi.stop()
