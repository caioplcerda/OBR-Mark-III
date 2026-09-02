# test_leds.py
# Exercita os padrões do LedController:
# - 7 LEDs sempre branco forte
# - 4+4 mostram status (cores fracas)
#
# Rode:  python3 test_leds.py

import time
from led_control import LedController

def wait(msg, t=1.2):
    print(msg)
    time.sleep(t)

def main():
    leds = None
    try:
        leds = LedController(pin=12, brightness=150)  # GPIO12 (pino físico 32)
        if not leds or not leds.enabled:
            print("LEDs indisponíveis: verifique instalação da lib rpi_ws281x e ligações.")
            return

        # Base: 7 branco forte; 4+4 branco fraco
        leds.status_ok_idle()
        wait("Base aplicada (7 branco forte; 4+4 branco fraco).")

        # OK (seguindo normal): barras azuis fracas
        leds.status_ok()
        wait("OK (azul fraco nas barras).")

        # AHEAD (interseção à frente): barras amarelas
        leds.status_ahead()
        wait("AHEAD (amarelo nas barras).")

        # INTERSECTION confirmada: barras âmbar
        leds.status_intersection()
        wait("INTERSECTION (âmbar nas barras).")

        # TURN left: esquerda verde
        leds.status_turn("left")
        wait("TURN LEFT (barras esquerdas verdes).")

        # TURN right: direita verde
        leds.status_turn("right")
        wait("TURN RIGHT (barras direitas verdes).")

        # U-TURN: roxo piscando nas barras
        leds.status_turn("uturn")
        wait("U-TURN (roxo piscando).")

        # LOST: vermelho piscando
        leds.status_lost()
        wait("LOST (vermelho piscando).")

        # FOLLOWING: efeito "wipe" azul nas barras
        print("FOLLOWING (wipe azul)…")
        for _ in range(3):
            leds.status_following()

        # Volta para base
        leds.status_ok_idle()
        wait("Voltando à base (7 branco, 4+4 branco fraco).", t=0.8)

        print("Teste concluído ✅")
    except KeyboardInterrupt:
        pass
    finally:
        if leds:
            try:
                # garante o estado base no encerramento
                leds.status_ok_idle()
                leds.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    main()
