from flask import Flask, request, render_template_string
import RPi.GPIO as GPIO
import time

app = Flask(__name__)

# Configuração inicial do GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Template HTML simples
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servo</title>
</head>
<body>
    <h1>Controle de Servo SG90</h1>
    <form method="POST" action="/set_pwm">
        <label for="gpio">GPIO (BCM):</label>
        <input type="number" id="gpio" name="gpio" required><br><br>

        <label for="pwm">PWM (duty cycle 2.5 a 12.5):</label>
        <input type="number" id="pwm" name="pwm" step="0.1" required><br><br>

        <button type="submit">Enviar</button>
    </form>
</body>
</html>
"""

# Guardar servos ativos para não recriar PWM toda hora
servos = {}

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_pwm", methods=["POST"])
def set_pwm():
    gpio = int(request.form["gpio"])
    duty = float(request.form["pwm"])

    # Se não existir PWM ainda nesse pino, cria
    if gpio not in servos:
        GPIO.setup(gpio, GPIO.OUT)
        pwm = GPIO.PWM(gpio, 50)  # SG90 = 50Hz
        pwm.start(0)
        servos[gpio] = pwm
    else:
        pwm = servos[gpio]

    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)  # tempo para o servo ir até posição
    pwm.ChangeDutyCycle(0)  # para não ficar tremendo

    return f"Servo no GPIO {gpio} ajustado para duty {duty}"

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        for pwm in servos.values():
            pwm.stop()
        GPIO.cleanup()
