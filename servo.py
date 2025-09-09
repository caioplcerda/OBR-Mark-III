from flask import Flask, request, render_template_string
from gpiozero import AngularServo
from time import sleep

app = Flask(__name__)

html_template = """..."""  # seu HTML permanece igual

servos = {}

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    if gpio not in servos:
        servo = AngularServo(gpio, min_angle=0, max_angle=180)
        servos[gpio] = servo
    else:
        servo = servos[gpio]

    servo.angle = angle
    sleep(0.5)

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
