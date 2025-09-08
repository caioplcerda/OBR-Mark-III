from flask import Flask, request, render_template_string
from gpiozero import Servo
from time import sleep

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servo</title>
</head>
<body>
    <h1>Controle de Servo SG90</h1>
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

def angle_to_value(angle):
    # Converte 0-180° para valor -1 a 1 do gpiozero
    return (angle / 90) - 1

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    if gpio not in servos:
        servo = Servo(gpio)
        servos[gpio] = servo
    else:
        servo = servos[gpio]

    servo.value = angle_to_value(angle)
    sleep(0.5)
    servo.detach()  # evita tremores

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
