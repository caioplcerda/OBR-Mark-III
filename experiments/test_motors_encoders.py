# test_motors_encoders.py
# Teste isolado de motores (TB6612FNG, canal B) no Raspberry Pi, sem encoders.
# - Esquerda:  BIN1=17, BIN2=27, PWMB=22
# - Direita:   BIN1=23, BIN2=24, PWMB=25 (inversão lógica para frente)
# - STBY:      5
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

# ===== CONFIG TESTE =====
DUTY = 100          # % PWM
WINDOW = 5.0       # segundos cada etapa
PAUSE = 0.6        # pausa entre etapas

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

def run_window(h, set_dir_fn, desc, left_duty, right_duty):
    """Roda uma janela de teste (5s) com direções específicas."""
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
    print("Motores executados. Verifique movimento físico.")
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

    # Liga driver
    lgpio.gpio_write(h, STBY, 1)

    try:
        print("== TESTE MOTORES (SEM ENCODERS) ==")
        print(f"Duty = {DUTY}% | Janela = {WINDOW:.1f}s\n")

        # 1. Ambos frente
        def both_forward(h):
            motor_left_forward(h)
            motor_right_forward(h)
        run_window(h, both_forward, "Ambos frente", DUTY, DUTY)
        time.sleep(PAUSE)

        # 2. Ambos trás
        def both_reverse(h):
            motor_left_reverse(h)
            motor_right_reverse(h)
        run_window(h, both_reverse, "Ambos trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 3. Esquerdo frente, direito trás
        def left_forward_right_reverse(h):
            motor_left_forward(h)
            motor_right_reverse(h)
        run_window(h, left_forward_right_reverse, "Esq frente, Dir trás", DUTY, DUTY)
        time.sleep(PAUSE)

        # 4. Esquerdo trás, direito frente
        def left_reverse_right_forward(h):
            motor_left_reverse(h)
            motor_right_forward(h)
        run_window(h, left_reverse_right_forward, "Esq trás, Dir frente", DUTY, DUTY)
        time.sleep(PAUSE)

        print("OBS: Sem encoders, verifique fisicamente se 'frente' move o robô para frente.")
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
            lgpio.gpiochip_close(h)
        except Exception:
            pass

        print("Encerrado.")

if __name__ == "__main__":
    main()
