from flask import Flask, request, render_template_string
import pigpio
from time import sleep

app = Flask(__name__)

# HTML do formulário
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

# Inicializa pigpio
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Não foi possível conectar ao daemon pigpio. Execute 'sudo pigpiod'.")

# Armazena servos já usados
servos = {}

# Converte ângulo 0–180 para pulso 500–2500 µs
def angle_to_pulse(angle):
    return 500 + (angle / 180) * 2000

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_angle", methods=["POST"])
def set_angle():
    gpio = int(request.form["gpio"])
    angle = float(request.form["angle"])

    # Ativa o servo se ainda não estiver ativado
    if gpio not in servos:
        servos[gpio] = True  # apenas marca que foi usado

    pulse = angle_to_pulse(angle)
    pi.set_servo_pulsewidth(gpio, pulse)
    sleep(0.5)  # tempo para o servo se mover

    return f"Servo no GPIO {gpio} ajustado para {angle}°"

# Desliga todos os servos ao fechar o servidor
@app.route("/shutdown", methods=["GET"])
def shutdown():
    for gpio in servos:
        pi.set_servo_pulsewidth(gpio, 0)
    pi.stop()
    return "Servos desligados e daemon pigpio parado."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
