from flask import Flask, request, render_template_string
import pigpio
from time import sleep
import atexit

app = Flask(__name__)

# HTML do formulário
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servo (PWM)</title>
</head>
<body>
    <h1>Controle de Servo por PWM</h1>
    <form method="POST" action="/set_pwm">
        <label for="gpio">GPIO (BCM):</label>
        <input type="number" id="gpio" name="gpio" required><br><br>

        <label for="pwm">Valor PWM (500-2500):</label>
        <input type="number" id="pwm" name="pwm" min="500" max="2500" required><br><br>

        <button type="submit">Enviar</button>
    </form>
    <hr>
    <form method="POST" action="/shutdown">
        <button type="submit">Desligar Todos os Servos</button>
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

def cleanup():
    """Desliga todos os servos que foram ativados."""
    print("Desligando servos...")
    for gpio in servos:
        pi.set_servo_pulsewidth(gpio, 0)
    servos.clear()
    print("Servos desligados.")

# Registra a função de limpeza para ser chamada ao sair
atexit.register(cleanup)

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/set_pwm", methods=["POST"])
def set_pwm():
    gpio = int(request.form["gpio"])
    pwm = int(request.form["pwm"])

    # Ativa o servo se ainda não estiver ativado
    if gpio not in servos:
        servos[gpio] = True  # apenas marca que foi usado

    pi.set_servo_pulsewidth(gpio, pwm)
    sleep(0.5)  # tempo para o servo se mover

    return f"Servo no GPIO {gpio} ajustado para PWM {pwm}"

# Desliga todos os servos ao fechar o servidor
@app.route("/shutdown", methods=["POST"])
def shutdown():
    cleanup()
    return "Todos os servos foram desligados."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
