# test_motors_encoders_v2.py
# Teste isolado de motores + encoders (TB6612FNG, canal B) no Raspberry Pi.
# - Esquerda:  BIN1=17, BIN2=27, PWMB=22
# - Direita:   BIN1=23, BIN2=24, PWMB=25 (fiação comum costuma exigir inversão lógica)
# - STBY:      5
# Encoders (canal A): Esq=BCM 6, Dir=BCM 19. (B estão listados se quiser direção no futuro)
#
# Roda cada motor sozinho (frente e ré), por 2s, e mostra os ticks.

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
WINDOW = 5.0      # segundos cada etapa
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
    # Muitos chassis precisam desta lógica p/ "frente" no direito:
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

def run_window(pwm_l, pwm_r, set_dir_fn, which):
    """Roda uma janela de teste (2s) para um motor específico."""
    global ticks_l, ticks_r
    ticks_l = 0
    ticks_r = 0

    stop_motors()
    set_dir_fn()
    pwm_l.ChangeDutyCycle(DUTY)
    pwm_r.ChangeDutyCycle(DUTY)
    t0 = time.time()
    while time.time() - t0 < WINDOW:
        time.sleep(0.05)

    # Para o PWM do motor não testado (mantém só o testado ativo na janela)
    if which == "L":
        pwm_r.ChangeDutyCycle(0)
    else:
        pwm_l.ChangeDutyCycle(0)
    stop_motors()

    return ticks_l, ticks_r

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

        # ----- ESQUERDA FRENTE -----
        print("[L] Frente...")
        tl, tr = run_window(pwm_left, pwm_right, motor_left_forward, "L")
        print(f"Ticks (L,R): {tl}, {tr}")
        if tl == 0:
            print("  ! Encoder ESQUERDO não contou na frente. Verifique fio do ENCODER_A_L (BCM 6), GND e VCC.\n")
        else:
            print("  ✓ Encoder ESQUERDO OK na frente.\n")
        time.sleep(PAUSE)

        # ----- ESQUERDA RÉ -----
        print("[L] Ré...")
        tl, tr = run_window(pwm_left, pwm_right, motor_left_reverse, "L")
        print(f"Ticks (L,R): {tl}, {tr}")
        if tl == 0:
            print("  ! Encoder ESQUERDO não contou na ré. Se contou na frente, talvez a roda não girou na ré (TB6612, fiação BIN1/BIN2) ou atrito mecânico.\n")
        else:
            print("  ✓ Encoder ESQUERDO OK na ré.\n")
        time.sleep(PAUSE)

        # ----- DIREITA FRENTE -----
        print("[R] Frente...")
        tl, tr = run_window(pwm_left, pwm_right, motor_right_forward, "R")
        print(f"Ticks (L,R): {tl}, {tr}")
        if tr == 0:
            print("  ! Encoder DIREITO não contou na frente. Verifique ENCODER_A_R (BCM 19), GND, VCC.\n")
        else:
            print("  ✓ Encoder DIREITO OK na frente.\n")
        time.sleep(PAUSE)

        # ----- DIREITA RÉ -----
        print("[R] Ré...")
        tl, tr = run_window(pwm_left, pwm_right, motor_right_reverse, "R")
        print(f"Ticks (L,R): {tl}, {tr}")
        if tr == 0:
            print("  ! Encoder DIREITO não contou na ré. Se contou na frente, talvez a roda não girou na ré (TB6612, fiação BIN1/BIN2) ou atrito mecânico.\n")
        else:
            print("  ✓ Encoder DIREITO OK na ré.\n")
        time.sleep(PAUSE)

        print("OBS: Com encoder de 1 canal estamos só contando pulsos (sem sentido).")
        print("     Para checar inversão 'frente/ré', observe fisicamente o giro da roda.")
        print("     Se o direito gira ao contrário na frente, inverta a lógica do R_BIN1/R_BIN2 no seu código.")

        # Loop live (Ctrl+C para sair)
        print("\n[LIVE] Girando manualmente as rodas deve aumentar os ticks.")
        while True:
            print(f"[LIVE] Ticks L={ticks_l} | R={ticks_r}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        # Para tudo com cuidado para evitar erros no __del__
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
