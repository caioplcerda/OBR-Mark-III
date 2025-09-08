from flask import Flask, request, render_template_string
from gpiozero import Servo
from time import sleep

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Servo Control</title>
</head>
<body>
    <h1>SG90 Servo Control with PWM</h1>
    <form method="POST" action="/set_pulse_width">
        <label for="gpio">GPIO (BCM):</label>
        <input type="number" id="gpio" name="gpio" required><br><br>

        <label for="pulse_width">Pulse Width (ms):</label>
        <input type="number" id="pulse_width" name="pulse_width" min="0.5" max="2.5" step="0.1" required><br><br>

        <button type="submit">Set Pulse Width</button>
    </form>
</body>
</html>
"""

servos = {}

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_pulse_width", methods=["POST"])
def set_pulse_width():
    gpio = int(request.form["gpio"])
    pulse_width = float(request.form["pulse_width"])

    if gpio not in servos:
        # SG90 datasheet says pulse width is 0.5ms to 2.4ms, but we'll use a slightly wider range
        # to be safe. Gpiozero expects pulse width in seconds.
        servo = Servo(gpio, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
        servos[gpio] = servo
    else:
        servo = servos[gpio]

    servo.pulse_width = pulse_width / 1000  # Convert ms to seconds
    sleep(0.5)
    servo.detach()  # avoids jitter

    return f"Servo on GPIO {gpio} set to {pulse_width}ms pulse width"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
