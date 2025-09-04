# test_leds_spi.py
# Três módulos WS2812 (7 LEDs cada) em série (DIN->DOUT->DOUT).
# Mapeamento solicitado:
#   seg 0 = ESQUERDA   -> pixels  0.. 6
#   seg 1 = DIREITA    -> pixels  7..13
#   seg 2 = MEIO       -> pixels 14..20
#
# Backend:
#   1) Preferência: SPI (neopixel_spi) em SPI0 MOSI (GPIO10 / pino 19)
#   2) Fallback: rpi_ws281x (PWM) em GPIO12 (pino 32)
#
# Execute de preferência com sudo para rpi_ws281x:
#   sudo python3 test_leds_spi.py

import time
import sys

LED_COUNT = 21  # 3 x 7

# ---------- BACKENDS ----------
BACKEND = None  # "SPI" ou "WS281X"

# SPI (opcional)
_SPI_OK = False
try:
    # Adafruit CircuitPython neopixel_spi
    # pip: adafruit-circuitpython-neopixel-spi
    import busio
    import board
    import neopixel_spi as neopixel
    _SPI_OK = True
except Exception:
    _SPI_OK = False

# rpi_ws281x (fallback por PWM)
_WS_OK = False
try:
    from rpi_ws281x import Adafruit_NeoPixel, Color
    _WS_OK = True
except Exception:
    _WS_OK = False


# ---------- CONFIG ----------
# SPI: usa SPI0 MOSI (GPIO10 / pino físico 19), SCK (GPIO11 / pino 23)
# brilho (0.0..1.0) no backend SPI; no WS281X é (0..255)
SPI_BRIGHTNESS = 0.6

# WS281X: PWM em GPIO12 (pino 32)
WS_PIN         = 12
WS_FREQ_HZ     = 800000
WS_DMA         = 10
WS_INVERT      = False
WS_BRIGHTNESS  = 160  # 0..255

# Ordem e nomes
SEGMENT_NAMES = ["esquerda", "direita", "meio"]

# Se algum bloco está fisicamente invertido (fio ao contrário), marque True
# Índices: 0=esquerda, 1=direita, 2=meio
REVERSE_SEGMENT = [False, False, False]


# ---------- ABSTRAÇÃO SIMPLES ----------
class StripBase:
    def show(self): ...
    def set_pixel_rgb(self, i, r, g, b): ...
    def blackout(self): ...


class StripSPI(StripBase):
    def __init__(self, count, brightness=SPI_BRIGHTNESS):
        # SPI0 padrão do Pi
        # - board.SPI() já cria em SCK=GPIO11, MOSI=GPIO10
        self.spi = busio.SPI(board.SCK, MOSI=board.MOSI)
        # `neopixel.NeoPixel_SPI` assume ordem GRB
        self.pixels = neopixel.NeoPixel_SPI(self.spi, count, brightness=brightness, auto_write=False)
        self.count = count

    def show(self):
        self.pixels.show()

    def set_pixel_rgb(self, i, r, g, b):
        if 0 <= i < self.count:
            # CircuitPython usa ordem RGB
            self.pixels[i] = (int(r), int(g), int(b))

    def blackout(self):
        for i in range(self.count):
            self.pixels[i] = (0, 0, 0)
        self.pixels.show()


class StripWS(StripBase):
    def __init__(self, count):
        self.strip = Adafruit_NeoPixel(
            count, WS_PIN, WS_FREQ_HZ, WS_DMA, WS_INVERT, WS_BRIGHTNESS
        )
        self.strip.begin()
        self.count = count

    def show(self):
        self.strip.show()

    def set_pixel_rgb(self, i, r, g, b):
        if 0 <= i < self.count:
            self.strip.setPixelColor(i, Color(int(r), int(g), int(b)))

    def blackout(self):
        for i in range(self.count):
            self.strip.setPixelColor(i, 0)
        self.strip.show()


def make_strip():
    global BACKEND
    if _SPI_OK:
        try:
            s = StripSPI(LED_COUNT, brightness=SPI_BRIGHTNESS)
            BACKEND = "SPI"
            print("[LED] Backend: SPI (neopixel_spi)")
            return s
        except Exception as e:
            print("[LED] Falha SPI:", e)
    if _WS_OK:
        try:
            s = StripWS(LED_COUNT)
            BACKEND = "WS281X"
            print("[LED] Backend: rpi_ws281x (PWM GPIO12)")
            return s
        except Exception as e:
            print("[LED] Falha rpi_ws281x:", e)
    print("[LED] Nenhum backend disponível. Instale 'adafruit-circuitpython-neopixel-spi' (SPI) ou 'rpi_ws281x'.")
    sys.exit(1)


# ---------- UTIL ----------
def seg_bounds(seg_index: int):
    """Faixa inclusiva de 7 LEDs por segmento."""
    start = seg_index * 7
    end = start + 6
    return start, end

def set_segment(strip: StripBase, seg_index: int, r: int, g: int, b: int):
    """Pinta um segmento inteiro (7 LEDs) com (r,g,b)."""
    start, end = seg_bounds(seg_index)
    rng = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        rng = list(reversed(rng))
    for i in rng:
        strip.set_pixel_rgb(i, r, g, b)

def white(strip, seg_index: int, level: int):
    level = max(0, min(255, level))
    set_segment(strip, seg_index, level, level, level)

def blackout(strip):
    strip.blackout()

def demo_identify(strip):
    """Liga cores distintas por bloco para conferir o mapeamento solicitado."""
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
    r,g,b = color
    start, end = seg_bounds(seg_index)
    seq = list(range(start, end+1))
    if REVERSE_SEGMENT[seg_index]:
        seq = list(reversed(seq))
    for _ in range(loops):
        for i in range(len(seq)):
            # apaga bloco
            for j in seq:
                strip.set_pixel_rgb(j, 0, 0, 0)
            # acende a "cabeça"
            strip.set_pixel_rgb(seq[i], r, g, b)
            strip.show()
            time.sleep(delay)

def main():
    strip = make_strip()
    blackout(strip)

    try:
        print("== Teste LEDs (3 módulos x 7) ==")
        print("Ordem:", SEGMENT_NAMES)
        print("Reverse:", REVERSE_SEGMENT)
        print("Backend:", BACKEND)

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

        # Passo 4: chase por bloco (checar orientação)
        for idx, name in enumerate(SEGMENT_NAMES):
            print(f"- Chase {name}")
            demo_chase(strip, idx, loops=2, delay=0.06, color=(255,255,255))
            time.sleep(0.25)

        print("Se algum bloco correu ao contrário, marque REVERSE_SEGMENT[índice]=True.")
        print("Ctrl+C para sair; deixo fraco no final.")
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
