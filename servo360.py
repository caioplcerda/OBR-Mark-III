from gpiozero import Servo
from time import sleep

servo = Servo(18)  # change to your GPIO pin

try:
    print("Moving to min")
    servo.min()
    sleep(1)

    print("Moving to mid")
    servo.mid()
    sleep(1)

    print("Moving to max")
    servo.max()
    sleep(1)

finally:
    print("Test complete")
