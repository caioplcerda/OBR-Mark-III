# test_motors_encoders.py
# Teste isolado de motores + encoders (TB6612FNG, canal B) no Raspberry Pi.
# - Esquerda:  BIN1=17, BIN2=27, PWMB=22
# - Direita:   BIN1=23, BIN2=24, PWMB=25 (inversão lógica para frente)
# - STBY:      5
# Encoders (canal A): Esq=BCM 6, Dir=BCM 19
#
# Testes:
# 1. Ambos motores frente (5s)
# 2. Ambos motores trás (5s)
# 3. Esquerdo frente, direito trás (5s)
# 4. Esquerdo trás, direito frente (5s)

import RPi.GPIO as GPIO
import time

# ===== PINAGEM MOTOR =====
L_BIN1, L_BIN2, L_PWMB = 17, 27, 22
R_BIN1, R_BIN2, R_PWMB = 23, 24, 25
STBY = 5

# ===== ENCODERS (canal A usado p/ contar) =====
ENCODER_A_L, ENCODER_B_L = 6, 13
ENCODER_A_R, ENCODER_B_R = 19, 26

# ===== CONFIG TESTE =====
DUTY = 100          # % PWM
WINDOW = 5.0       # segundos cada etapa
PAUSE = 0.6        # pausa entre etapas

# ===== CONTADORES =====
ticks_l = 0
ticks_r = 0

def enc_l(_):
    global ticks_l
    ticks_l += 1

def enc_r(_):
    global ticks_r
    ticks_r += 1

def motor_left_forward():
    GPIO.output(L_BIN1, GPIO.HIGH)
    GPIO.output(L_BIN2, GPIO.LOW)

def motor_left_reverse():
    GPIO.output(L_BIN1, GPIO.LOW)
    GPIO.output(L_BIN2, GPIO.HIGH)

def motor_right_forward():
    # Inversão lógica para "frente" no direito (chassi espelhado)
    GPIO.output(R_BIN1, GPIO.HIGH)
    GPIO.output(R_BIN2, GPIO.LOW)

def motor_right_reverse():
    GPIO.output(R_BIN1, GPIO.LOW)
    GPIO.output(R_BIN2, GPIO.HIGH)

def stop_motors():
    GPIO.output(L_BIN1, GPIO.LOW)
    GPIO.output(L_BIN2, GPIO.LOW)
    GPIO.output(R_BIN1, GPIO.LOW)
    GPIO.output(R_BIN2, GPIO.LOW)

def run_window(pwm_l, pwm_r, set_dir_fn, desc, left_duty, right_duty):
    """Roda uma janela de teste (5s) com direções específicas."""
    global ticks_l, ticks_r
    ticks_l = 0
    ticks_r = 0

    stop_motors()
    set_dir_fn()
    pwm_l.ChangeDutyCycle(left_duty)
    pwm_r.ChangeDutyCycle(right_duty)
    t0 = time.time()
    while time.time() - t0 < WINDOW:
        time.sleep(0.05)

    stop_motors()
    pwm_l.ChangeDutyCycle(0)
    pwm_r.ChangeDutyCycle(0)

    print(f"[TESTE] {desc}")
    print(f"Ticks (L,R): {ticks_l}, {ticks_r}")
    if ticks_l == 0 and left_duty != 0:
        print("  ! Encoder ESQUERDO não contou. Verifique BCM 6, GND, VCC ou fiação BIN1/BIN2.")
    if ticks_r == 0 and right_duty != 0:
        print("  ! Encoder DIREITO não contou. Verifique BCM 19, GND, VCC ou fiação BIN1/BIN2.")
    if ticks_l > 0 and left_duty != 0:
        print("  ✓ Encoder ESQUERDO OK.")
    if ticks_r > 0 and right_duty != 0:
        print("  ✓ Encoder DIREITO OK.")
    print()

def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    # Motores
    for pin in [L_BIN1, L_BIN2, R_BIN1, R_BIN2, STBY]:
        GPIO.setup(pin, GPIO.OUT)
    GPIO.setup(L_PWMB, GPIO.OUT)
    GPIO.setup(R_PWMB, GPIO.OUT)
    pwm_left = GPIO.PWM(L_PWMB, 1000)
    pwm_right = GPIO.PWM(R_PWMB, 1000)
    pwm_left.start(0)
    pwm_right.start(0)

    # Encoders
    GPIO.setup(ENCODER_A_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCODER_A_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(ENCODER_A_L, GPIO.RISING, callback=enc_l)
    GPIO.add_event_detect(ENCODER_A_R, GPIO.RISING, callback=enc_r)

    # Liga driver
    GPIO.output(STBY, GPIO.HIGH)

    try:
        print("== TESTE MOTORES + ENCODERS ==")
        print(f"Duty = {DUTY}% | Janela = {WINDOW:.1f}s\n")

        # 1. Ambos frente
        def both_forward():
            motor_left_forward()
            motor_right_forward()
        run_window(pwm_left, pwm_right, both_forward, "Ambos frente", DUTY, DUTY)
        time.sleep(PAUSE)

        # 2. Ambos trás
        def both_reverse():
            motor_left_reverse()
            motor_right_reverse()
        run_window(pwm_left, pwm_right, both_reverse, "Ambos trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 3. Esquerdo frente, direito trás
        def left_forward_right_reverse():
            motor_left_forward()
            motor_right_reverse()
        run_window(pwm_left, pwm_right, left_forward_right_reverse, "Esq frente, Dir trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 4. Esquerdo trás, direito frente
        def left_reverse_right_forward():
            motor_left_reverse()
            motor_right_forward()
        run_window(pwm_left, pwm_right, left_reverse_right_forward, "Esq trás, Dir frente", DUTY, DUTY)
        time.sleep(PAUSE)

        print("OBS: Encoders de 1 canal contam pulsos (sem sentido).")
        print("     Cheque fisicamente se 'frente' move o robô para frente.")
        print("     Se o direito gira ao contrário, confirme inversão em R_BIN1/R_BIN2.")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_motors()
            pwm_left.ChangeDutyCycle(0)
            pwm_right.ChangeDutyCycle(0)
            GPIO.output(STBY, GPIO.LOW)
        except Exception:
            pass

        try:
            GPIO.remove_event_detect(ENCODER_A_L)
            GPIO.remove_event_detect(ENCODER_A_R)
        except Exception:
            pass

        try:
            pwm_left.stop()
            pwm_right.stop()
        except Exception:
            pass

        try:
            GPIO.cleanup()
        except Exception:
            pass

        print("Encerrado.")

if __name__ == "__main__":
    main()
