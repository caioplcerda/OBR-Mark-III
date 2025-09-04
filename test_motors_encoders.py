# test_motors_encoders.py
# Teste isolado de motores + encoders (TB6612FNG, canal B) no Raspberry Pi.
# Usa lgpio (alinhado com hardware_control.py).
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

import lgpio
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

def enc_l(_chip, _gpio, level, _tick):
    global ticks_l
    if level == 1:
        ticks_l += 1

def enc_r(_chip, _gpio, level, _tick):
    global ticks_r
    if level == 1:
        ticks_r += 1

def motor_left_forward(h):
    lgpio.gpio_write(h, L_BIN1, 1)
    lgpio.gpio_write(h, L_BIN2, 0)

def motor_left_reverse(h):
    lgpio.gpio_write(h, L_BIN1, 0)
    lgpio.gpio_write(h, L_BIN2, 1)

def motor_right_forward(h):
    # Inversão lógica para "frente" no direito (chassi espelhado)
    lgpio.gpio_write(h, R_BIN1, 1)
    lgpio.gpio_write(h, R_BIN2, 0)

def motor_right_reverse(h):
    lgpio.gpio_write(h, R_BIN1, 0)
    lgpio.gpio_write(h, R_BIN2, 1)

def stop_motors(h):
    lgpio.gpio_write(h, L_BIN1, 0)
    lgpio.gpio_write(h, L_BIN2, 0)
    lgpio.gpio_write(h, R_BIN1, 0)
    lgpio.gpio_write(h, R_BIN2, 0)

def run_window(h, pwm_l, pwm_r, set_dir_fn, desc, left_duty, right_duty):
    """Roda uma janela de teste (5s) com direções específicas."""
    global ticks_l, ticks_r
    ticks_l = 0
    ticks_r = 0

    stop_motors(h)
    set_dir_fn(h)
    lgpio.tx_pwm(h, L_PWMB, 1000, left_duty)
    lgpio.tx_pwm(h, R_PWMB, 1000, right_duty)
    t0 = time.time()
    while time.time() - t0 < WINDOW:
        time.sleep(0.05)

    stop_motors(h)
    lgpio.tx_pwm(h, L_PWMB, 1000, 0)
    lgpio.tx_pwm(h, R_PWMB, 1000, 0)

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
    # Inicializa lgpio
    h = lgpio.gpiochip_open(0)
    if h < 0:
        print("Erro: Não conseguiu abrir o chip GPIO. Verifique permissões (sudo) ou instale liblgpio.")
        return

    # Configura pinos motores
    for pin in [L_BIN1, L_BIN2, R_BIN1, R_BIN2, STBY]:
        lgpio.gpio_claim_output(h, pin, 0)
    lgpio.gpio_claim_output(h, L_PWMB, 0)
    lgpio.gpio_claim_output(h, R_PWMB, 0)

    # Configura encoders
    lgpio.gpio_claim_input(h, ENCODER_A_L, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_input(h, ENCODER_A_R, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_alert(h, ENCODER_A_L, lgpio.RISING_EDGE, 0)
    lgpio.gpio_claim_alert(h, ENCODER_A_R, lgpio.RISING_EDGE, 0)
    lgpio.gpio_set_alert_func(h, ENCODER_A_L, enc_l)
    lgpio.gpio_set_alert_func(h, ENCODER_A_R, enc_r)

    # Liga driver
    lgpio.gpio_write(h, STBY, 1)

    try:
        print("== TESTE MOTORES + ENCODERS ==")
        print(f"Duty = {DUTY}% | Janela = {WINDOW:.1f}s\n")

        # 1. Ambos frente
        def both_forward(h):
            motor_left_forward(h)
            motor_right_forward(h)
        run_window(h, None, None, both_forward, "Ambos frente", DUTY, DUTY)
        time.sleep(PAUSE)

        # 2. Ambos trás
        def both_reverse(h):
            motor_left_reverse(h)
            motor_right_reverse(h)
        run_window(h, None, None, both_reverse, "Ambos trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 3. Esquerdo frente, direito trás
        def left_forward_right_reverse(h):
            motor_left_forward(h)
            motor_right_reverse(h)
        run_window(h, None, None, left_forward_right_reverse, "Esq frente, Dir trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 4. Esquerdo trás, direito frente
        def left_reverse_right_forward(h):
            motor_left_reverse(h)
            motor_right_forward(h)
        run_window(h, None, None, left_reverse_right_forward, "Esq trás, Dir frente", DUTY, DUTY)
        time.sleep(PAUSE)

        print("OBS: Encoders de 1 canal contam pulsos (sem sentido).")
        print("     Cheque fisicamente se 'frente' move o robô para frente.")
        print("     Se o direito gira ao contrário, confirme inversão em R_BIN1/R_BIN2.")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_motors(h)
            lgpio.tx_pwm(h, L_PWMB, 1000, 0)
            lgpio.tx_pwm(h, R_PWMB, 1000, 0)
            lgpio.gpio_write(h, STBY, 0)
        except Exception:
            pass

        try:
            lgpio.gpio_set_alert_func(h, ENCODER_A_L, None)
            lgpio.gpio_set_alert_func(h, ENCODER_A_R, None)
        except Exception:
            pass

        try:
            lgpio.gpiochip_close(h)
        except Exception:
            pass

        print("Encerrado.")

if __name__ == "__main__":
    main()
