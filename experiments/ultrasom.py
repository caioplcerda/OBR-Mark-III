import lgpio
import time
from flask import Flask, render_template_string

# GPIO pins
TRIG = 12
ECHO = 6
STATE_PIN = 13

# Open GPIO chip
h = lgpio.gpiochip_open(0)

# Setup pins
lgpio.gpio_claim_output(h, TRIG)
lgpio.gpio_claim_input(h, ECHO)
lgpio.gpio_claim_input(h, STATE_PIN)

# Flask setup
app = Flask(__name__)

# HTML template
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
    # Ensure trigger low
    lgpio.gpio_write(h, TRIG, 0)
    time.sleep(0.0002)

    # Send 10µs pulse
    lgpio.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, TRIG, 0)

    # Wait for echo start
    while lgpio.gpio_read(h, ECHO) == 0:
        pulse_start = time.time()

    # Wait for echo end
    while lgpio.gpio_read(h, ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # cm
    return round(distance, 2)

@app.route('/')
def index():
    try:
        distance = read_distance()
    except Exception:
        distance = "Error"

    state = "HIGH" if lgpio.gpio_read(h, STATE_PIN) else "LOW"
    return render_template_string(HTML_TEMPLATE, distance=distance, state=state)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        lgpio.gpiochip_close(h)
    finally:
        lgpio.gpiochip_close(h)
