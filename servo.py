from flask import Flask, request, render_template_string
from gpiozero import AngularServo
from time import sleep

app = Flask(__name__)

html_template = """..."""  # mesmo HTML

servos = {}

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    if gpio not in servos:
        # calibrando pulso mínimo e máximo
        servo = AngularServo(
            gpio,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,  # ajuste se necessário
            max_pulse_width=0.0025   # ajuste se necessário
        )
        servos[gpio] = servo
    else:
        servo = servos[gpio]

    servo.angle = angle
    sleep(0.5)
    # servo.detach()  # opcional, só se quiser soltar

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
