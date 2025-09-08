from flask import Flask, render_template_string, request
import pigpio

app = Flask(__name__)
pi = pigpio.pi()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Controle de Servos</title>
</head>
<body>
    <h1>Controle de Servos - Raspberry Pi 5</h1>
    <form method="POST">
        <label>GPIO:</label>
        <input type="number" name="gpio" required><br><br>
        <label>PWM (µs, ex: 1500 = centro):</label>
        <input type="number" name="pwm" required><br><br>
        <input type="submit" value="Enviar">
    </form>
    <p>{{msg}}</p>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    msg = ""
    if request.method == "POST":
        gpio = int(request.form["gpio"])
        pwm = int(request.form["pwm"])
        try:
            pi.set_servo_pulsewidth(gpio, pwm)
            msg = f"GPIO {gpio} configurado com {pwm} µs"
        except Exception as e:
            msg = f"Erro: {e}"
    return render_template_string(html, msg=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
