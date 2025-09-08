from flask import Flask, request, render_template_string
import pigpio
import time

app = Flask(__name__)
pi = pigpio.pi()  # inicia daemon pigpio

# Template HTML
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servos</title>
</head>
<body>
    <h1>Controle de Servos SG90 e MG996R</h1>
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

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    # Converte ângulo em pulse width (µs)
    pulse = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(gpio, pulse)
    time.sleep(0.5)
    pi.set_servo_pulsewidth(gpio, 0)  # desliga servo para não tremer

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        # Desliga todos os servos
        for gpio in [16, 17, 18, 19]:  # coloque aqui os pinos dos seus servos
            pi.set_servo_pulsewidth(gpio, 0)
        pi.stop()
