import RPi.GPIO as GPIO
import time

# === Pinos do TB6612FNG (canal B de cada placa) ===
L_BIN1 = 17
L_BIN2 = 27
L_PWMB = 22

R_BIN1 = 23
R_BIN2 = 24
R_PWMB = 25

STBY = 5

# === Encoders ===
ENCODER_A_L = 6
ENCODER_B_L = 13
ENCODER_A_R = 19
ENCODER_B_R = 26

# Contadores
ticks_l = 0
ticks_r = 0

def enc_l(channel):
    global ticks_l
    ticks_l += 1

def enc_r(channel):
    global ticks_r
    ticks_r += 1


def main():
    global ticks_l, ticks_r
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup motores
    for pin in [L_BIN1, L_BIN2, R_BIN1, R_BIN2, STBY]:
        GPIO.setup(pin, GPIO.OUT)
    GPIO.setup(L_PWMB, GPIO.OUT)
    GPIO.setup(R_PWMB, GPIO.OUT)

    pwm_left = GPIO.PWM(L_PWMB, 100)
    pwm_right = GPIO.PWM(R_PWMB, 100)
    pwm_left.start(0)
    pwm_right.start(0)

    # Setup encoders
    GPIO.setup(ENCODER_A_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCODER_A_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(ENCODER_A_L, GPIO.RISING, callback=enc_l)
    GPIO.add_event_detect(ENCODER_A_R, GPIO.RISING, callback=enc_r)

    # Liga driver
    GPIO.output(STBY, GPIO.HIGH)

    try:
        print("== Teste de motores e encoders ==")
        # Motores para frente
        print("Frente 50% duty...")
        GPIO.output(L_BIN1, GPIO.HIGH)
        GPIO.output(L_BIN2, GPIO.LOW)
        GPIO.output(R_BIN1, GPIO.HIGH)
        GPIO.output(R_BIN2, GPIO.LOW)
        pwm_left.ChangeDutyCycle(50)
        pwm_right.ChangeDutyCycle(50)
        ticks_l = 0
        ticks_r = 0
        time.sleep(3)
        print(f"Ticks após frente: Esq={ticks_l}, Dir={ticks_r}")

        # Motores para trás
        print("Ré 50% duty...")
        GPIO.output(L_BIN1, GPIO.LOW)
        GPIO.output(L_BIN2, GPIO.HIGH)
        GPIO.output(R_BIN1, GPIO.LOW)
        GPIO.output(R_BIN2, GPIO.HIGH)
        pwm_left.ChangeDutyCycle(50)
        pwm_right.ChangeDutyCycle(50)
        ticks_l = 0
        ticks_r = 0
        time.sleep(3)
        print(f"Ticks após ré: Esq={ticks_l}, Dir={ticks_r}")

        print("Se na frente um motor girou para trás, ele está invertido 😉")

        while True:
            # Mostra contagem contínua (pode Ctrl+C para sair)
            print(f"[LIVE] Esq={ticks_l} | Dir={ticks_r}")
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        pwm_left.stop()
        pwm_right.stop()
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()
        print("Encerrado.")


if __name__ == "__main__":
    main()
