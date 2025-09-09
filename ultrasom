import RPi.GPIO as GPIO
import time
from flask import Flask, Response, render_template_string

# GPIO setup
GPIO.setmode(GPIO.BCM)

TRIG = 12
ECHO = 6
STATE_PIN = 13

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(STATE_PIN, GPIO.IN)

# Flask setup
app = Flask(__name__)

# HTML template with auto-refreshing stream
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HC-SR04 Live Data</title>
    <meta http-equiv="refresh" content="1">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        h1 { color: #333; }
        .data { font-size: 24px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>HC-SR04 Distance Sensor</h1>
    <div class="data">Distance: {{ distance }} cm</div>
    <div class="data">GPIO 13 State: {{ state }}</div>
</body>
</html>
"""


def read_distance():
    # Ensure trigger is low
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    # Send 10us pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Wait for echo start
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    # Wait for echo end
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # speed of sound (cm/s)
    distance = round(distance, 2)

    return distance


@app.route('/')
def index():
    try:
        distance = read_distance()
    except Exception:
        distance = "Error"

    state = "HIGH" if GPIO.input(STATE_PIN) else "LOW"
    return render_template_string(HTML_TEMPLATE, distance=distance, state=state)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        GPIO.cleanup()
    finally:
        GPIO.cleanup()
