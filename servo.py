from flask import Flask, request, render_template_string
from gpiozero import AngularServo
from time import sleep

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servo</title>
</head>
<body>
    <h1>Controle de Servo</h1>
    <form method="POST" action="/set_angle">
        <label for="gpio">GPIO (BCM):</label>
        <input type="number" id="gpio" name="gpio" required><br><br>

        <label for="angle">Ângulo (0 a 180):</label>
        <input type="number" id="angle" name="angle" min="0" max="180" required><br><br>

        <button type="submit">Enviar</button>
    </form>
</body>
</html>
"""

servos = {}

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    if gpio not in servos:
        # AngularServo permite setar o ângulo diretamente
        servo = AngularServo(gpio, min_angle=0, max_angle=180)
        servos[gpio] = servo
    else:
        servo = servos[gpio]

    servo.angle = angle  # seta o ângulo diretamente
    sleep(0.5)

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
