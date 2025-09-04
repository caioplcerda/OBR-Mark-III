# test_leds.py
# Três módulos WS2812 (7 LEDs cada) em série no MESMO fio (DIN->DOUT->DOUT).
# Mapeamento pedido:
#   seg 0 = ESQUERDA -> pixels  0.. 6
#   seg 1 = DIREITA  -> pixels  7..13
#   seg 2 = MEIO     -> pixels 14..20
#
# Biblioteca: rpi_ws281x (mesma de antes). Execute preferencialmente com sudo:
#   sudo python3 test_leds.py
#
# Se algum módulo estiver fisicamente invertido (efeito "chase" corre ao contrário),
# marque REVERSE_SEGMENT[índice] = True.

import time
import sys

try:
    from rpi_ws281x import Adafruit_NeoPixel, Color
    LIB_OK = True
except Exception as e:
    print("[LED] rpi_ws281x não disponível —", e)
    LIB_OK = False

# ======== CONFIGURAÇÃO GERAL ========
LED_COUNT      = 21      # 3 x 7
LED_PIN        = 12      # GPIO12 (pino físico 32) — mesmo de antes
LED_FREQ_HZ    = 800000
LED_DMA        = 10
LED_INVERT     = False
LED_BRIGHTNESS = 160     # 0-255 (global); ajuste se quiser mais/menos brilho

# Nomes por segmento (0=esquerda, 1=direita, 2=meio)
SEGMENT_NAMES = ["esquerda", "direita", "meio"]

# Se algum bloco está montado "ao contrário", marque True no respectivo índice
REVERSE_SEGMENT = [False, False, False]

# ======== AUXÍLIOS ========
def seg_bounds(seg_index: int):
    """Faixa inclusiva de 7 LEDs para o segmento."""
    start = seg_index * 7
    end   = start + 6
    return start, end

def set_segment(strip, seg_index: int, r: int, g: int, b: int):
    """Pinta um segmento inteiro (7 LEDs) com (r,g,b)."""
    start, end = seg_bounds(seg_index)
    idxs = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        idxs = list(reversed(idxs))
    c = Color(int(r), int(g), int(b))
    for i in idxs:
        strip.setPixelColor(i, c)

def white(strip, seg_index: int, level: int):
    """Branco com intensidade 'level' (0..255) em um segmento."""
    level = max(0, min(255, int(level)))
    set_segment(strip, seg_index, level, level, level)

def blackout(strip):
    for i in range(LED_COUNT):
        strip.setPixelColor(i, 0)
    strip.show()

def demo_identify(strip):
    """Liga cores diferentes para identificar os blocos na prática."""
    # esquerda = vermelho
    set_segment(strip, 0, 255, 0, 0)
    # direita = verde
    set_segment(strip, 1, 0, 255, 0)
    # meio = azul
    set_segment(strip, 2, 0, 0, 255)
    strip.show()
    print("Identificação: esquerda=VERMELHO, direita=VERDE, meio=AZUL")
    time.sleep(2.0)

def demo_chase(strip, seg_index: int, loops=2, delay=0.07, color=(255,255,255)):
    """Efeito 'correndo' dentro do segmento para checar orientação física."""
    r, g, b = color
    start, end = seg_bounds(seg_index)
    seq = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        seq = list(reversed(seq))
    for _ in range(loops):
        for i in range(len(seq)):
            # Apaga bloco
            for j in seq:
                strip.setPixelColor(j, 0)
            # Acende “cabeça”
            strip.setPixelColor(seq[i], Color(int(r), int(g), int(b)))
            strip.show()
            time.sleep(delay)

def main():
    if not LIB_OK:
        print("LEDs indisponíveis: verifique instalação da lib rpi_ws281x e ligações.")
        sys.exit(1)

    strip = Adafruit_NeoPixel(
        LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS
    )
    strip.begin()
    blackout(strip)

    try:
        print("== Teste: 3 módulos de 7 LEDs (21 no total) ==")
        print("Ordem (segmentos):", SEGMENT_NAMES)
        print("Reverse:", REVERSE_SEGMENT)

        # Passo 1: identificar blocos
        demo_identify(strip)

        # Passo 2: todos fracos (status base)
        for idx in range(3):
            white(strip, idx, 40)
        strip.show()
        time.sleep(1.2)

        # Passo 3: reforça um por vez
        for idx, name in enumerate(SEGMENT_NAMES):
            print(f"- Intensificando {name}...")
            white(strip, idx, 220)
            strip.show()
            time.sleep(1.0)
            white(strip, idx, 40)
            strip.show()
            time.sleep(0.3)

        # Passo 4: chase por bloco (checar orientação real)
        for idx, name in enumerate(SEGMENT_NAMES):
            print(f"- Chase {name}")
            demo_chase(strip, idx, loops=2, delay=0.06, color=(255,255,255))
            time.sleep(0.25)

        print("Se o 'chase' correu ao contrário em algum bloco, ajuste REVERSE_SEGMENT para True naquele índice.")
        print("Ctrl+C para sair; deixo todos fracos ao final.")

        # Mantém fraco no final (status base)
        for idx in range(3):
            white(strip, idx, 40)
        strip.show()

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        blackout(strip)
        time.sleep(0.05)

if __name__ == "__main__":
    main()
