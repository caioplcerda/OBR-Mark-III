import time

# ==============================
# Tentativa de importar GPIO
# Se estiver em PC, roda com Mock
# ==============================
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except (ImportError, RuntimeError):
    class MockGPIO:
        BCM = "BCM"
        IN = "IN"
        OUT = "OUT"
        LOW = 0
        HIGH = 1
        PUD_UP = "PUD_UP"

        def setmode(self, *args, **kwargs): pass
        def setwarnings(self, *args, **kwargs): pass
        def setup(self, *args, **kwargs): pass
        def input(self, *args, **kwargs): return self.HIGH
        def output(self, *args, **kwargs): pass
        def cleanup(self): pass

    GPIO = MockGPIO()
    print("⚠️ GPIO rodando em modo simulado (mock).")

# ==============================
# Pinos
# ==============================
# Motores
MOTOR_LEFT_FWD = 17
MOTOR_LEFT_BACK = 18
MOTOR_RIGHT_FWD = 22
MOTOR_RIGHT_BACK = 23

# Sensores
SENSOR_LEFT = 5
SENSOR_RIGHT = 6
SENSOR_FRONT = 13

# Botões
BUTTON_START = 19
BUTTON_STOP = 26

# ==============================
# Configuração dos pinos
# ==============================
def setup():
    # Motores
    GPIO.setup(MOTOR_LEFT_FWD, GPIO.OUT)
    GPIO.setup(MOTOR_LEFT_BACK, GPIO.OUT)
    GPIO.setup(MOTOR_RIGHT_FWD, GPIO.OUT)
    GPIO.setup(MOTOR_RIGHT_BACK, GPIO.OUT)

    # Sensores
    GPIO.setup(SENSOR_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SENSOR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SENSOR_FRONT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Botões
    GPIO.setup(BUTTON_START, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_STOP, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ==============================
# Controle dos Motores
# ==============================
def stop():
    GPIO.output(MOTOR_LEFT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_LEFT_BACK, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACK, GPIO.LOW)

def forward():
    GPIO.output(MOTOR_LEFT_FWD, GPIO.HIGH)
    GPIO.output(MOTOR_LEFT_BACK, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_FWD, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_BACK, GPIO.LOW)

def backward():
    GPIO.output(MOTOR_LEFT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_LEFT_BACK, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACK, GPIO.HIGH)

def turn_left():
    GPIO.output(MOTOR_LEFT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_LEFT_BACK, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_FWD, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_BACK, GPIO.LOW)

def turn_right():
    GPIO.output(MOTOR_LEFT_FWD, GPIO.HIGH)
    GPIO.output(MOTOR_LEFT_BACK, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_FWD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACK, GPIO.LOW)

# ==============================
# Lógica principal
# ==============================
def main():
    setup()
    running = False  # Estado do robô
    print("🤖 Robô pronto. Pressione START para iniciar.")

    try:
        while True:
            # Controle de start e stop
            if GPIO.input(BUTTON_START) == GPIO.LOW:
                running = True
                print("▶️ Robô iniciado.")
                time.sleep(0.5)  # Evita múltiplos cliques

            if GPIO.input(BUTTON_STOP) == GPIO.LOW:
                running = False
                stop()
                print("⏹️ Robô parado.")
                time.sleep(0.5)

            if running:
                # Leitura dos sensores
                left = GPIO.input(SENSOR_LEFT)
                right = GPIO.input(SENSOR_RIGHT)
                front = GPIO.input(SENSOR_FRONT)

                # ======= Lógica de movimento =======
                if front == GPIO.LOW:
                    print("🚧 Obstáculo à frente! Recuando...")
                    backward()
                    time.sleep(0.3)
                    stop()
                    time.sleep(0.1)
                    turn_right()
                    time.sleep(0.3)
                    stop()

                elif left == GPIO.LOW and right == GPIO.HIGH:
                    print("↪️ Curva para a esquerda")
                    turn_left()

                elif right == GPIO.LOW and left == GPIO.HIGH:
                    print("↩️ Curva para a direita")
                    turn_right()

                elif left == GPIO.LOW and right == GPIO.LOW:
                    print("⬆️ Linha detectada em ambos os lados, indo em frente")
                    forward()

                else:
                    print("❌ Linha perdida, parando")
                    stop()

                time.sleep(0.05)  # Pequeno atraso para estabilidade
            else:
                stop()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Encerrando o programa...")
        stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
