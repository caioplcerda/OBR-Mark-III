#!/usr/bin/env python3
# test_leds_spi.py
# Três módulos WS2812 (7 LEDs cada) em série (total 21) usando SPI (spidev).
# Mapeamento:
#   segmento 0: "esquerda" -> pixels  0..6
#   segmento 1: "direita"  -> pixels  7..13
#   segmento 2: "meio"     -> pixels 14..20
#
# Se algum bloco acende invertido, marque REVERSE_SEGMENT[seg] = True.
#
# Execução:
#   sudo python3 test_leds_spi.py

import time
import spidev

# ========= CONFIGURAÇÃO GERAL =========
LED_COUNT        = 21                 # 3 x 7
SEGMENT_SIZE     = 7
SEGMENT_ORDER    = ["esquerda", "direita", "meio"]  # rótulos
REVERSE_SEGMENT  = [False, False, False]

SPI_BUS          = 0
SPI_DEV          = 0
SPI_MAX_HZ       = 3_200_000
SPI_MODE         = 0

GLOBAL_BRIGHTNESS = 0.6
RESET_TRAILER_LEN = 100

# ========= ENCODER WS2812 por SPI =========
NIBBLE_0 = 0b1000
NIBBLE_1 = 0b1110

def _byte_to_nibbles(b: int) -> int:
    out = 0
    for i in range(8):
        bit = (b & (1 << (7 - i))) != 0
        nib = NIBBLE_1 if bit else NIBBLE_0
        out = (out << 4) | nib
    return out

def _pack32_to_bytes(x: int) -> bytes:
    return bytes([(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF])

def encode_ws2812_frame(rgb_bytes: bytes) -> bytes:
    buf = bytearray()
    for b in rgb_bytes:
        nib32 = _byte_to_nibbles(b)
        buf.extend(_pack32_to_bytes(nib32))
    buf.extend(b"\x00" * RESET_TRAILER_LEN)
    return bytes(buf)

# ========= FRAME BUFFER =========
def clamp8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else x)

class WS2812_SPI:
    def __init__(self, led_count: int):
        self.n = led_count
        self.buf = [(0, 0, 0)] * self.n
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEV)
        self.spi.max_speed_hz = SPI_MAX_HZ
        self.spi.mode = SPI_MODE

    def set_pixel(self, idx: int, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
        if not (0 <= idx < self.n):
            return
        if scale is None:
            scale = 1.0
        r = clamp8(int(r * scale))
        g = clamp8(int(g * scale))
        b = clamp8(int(b * scale))
        self.buf[idx] = (g, r, b)

    def fill(self, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
        for i in range(self.n):
            self.set_pixel(i, r, g, b, scale)

    def clear(self):
        self.fill(0, 0, 0, scale=1.0)

    def show(self):
        raw = bytearray()
        for (g, r, b) in self.buf:
            raw.extend((g & 0xFF, r & 0xFF, b & 0xFF))
        payload = encode_ws2812_frame(bytes(raw))
        self.spi.writebytes(payload)

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass

# ========= SEGMENTOS =========
def seg_bounds(seg_index: int):
    start = seg_index * SEGMENT_SIZE
    end = start + SEGMENT_SIZE - 1
    return start, end

def set_segment(strip: WS2812_SPI, seg_index: int, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
    start, end = seg_bounds(seg_index)
    rng = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        rng = list(reversed(rng))
    for i in rng:
        strip.set_pixel(i, r, g, b, scale=scale)

def white(strip: WS2812_SPI, seg_index: int, level: int):
    level = clamp8(level)
    set_segment(strip, seg_index, level, level, level, scale=1.0)

def blackout(strip: WS2812_SPI):
    strip.clear()
    strip.show()

def demo_identify(strip: WS2812_SPI):
    # esquerda = vermelho
    set_segment(strip, 0, 255, 0, 0, scale=1.0)
    # direita = verde
    set_segment(strip, 1, 0, 255, 0, scale=1.0)
    # meio = azul
    set_segment(strip, 2, 0, 0, 255, scale=1.0)
    strip.show()
    print("Identificação: esquerda=VERMELHO, direita=VERDE, meio=AZUL")
    time.sleep(2.0)

def demo_chase(strip: WS2812_SPI, seg_index: int, loops=2, delay=0.07, color=(255,255,255)):
    r, g, b = color
    start, end = seg_bounds(seg_index)
    seq = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        seq = list(reversed(seq))
    for _ in range(loops):
        for i in range(len(seq)):
            for j in seq:
                strip.set_pixel(j, 0, 0, 0, scale=1.0)
            strip.set_pixel(seq[i], r, g, b, scale=1.0)
            strip.show()
            time.sleep(delay)

# ========= MAIN =========
def main():
    print("== Teste WS2812 via SPI (3x7 LEDs) ==")
    print("SPI: /dev/spidev%d.%d @ %d Hz" % (SPI_BUS, SPI_DEV, SPI_MAX_HZ))
    print("Ordem lógica:", SEGMENT_ORDER)
    print("Reverse flags:", REVERSE_SEGMENT)

    strip = WS2812_SPI(LED_COUNT)

    try:
        blackout(strip)
        demo_identify(strip)

        for seg in range(3):
            white(strip, seg, 40)
        strip.show()
        time.sleep(1.2)

        for seg, name in enumerate(SEGMENT_ORDER):
            print(f"Fortalecendo {name}...")
            white(strip, seg, 220)
            strip.show()
            time.sleep(1.2)
            white(strip, seg, 40)
            strip.show()
            time.sleep(0.3)

        for seg, name in enumerate(SEGMENT_ORDER):
            print(f"Chase em {name}...")
            demo_chase(strip, seg, loops=2, delay=0.06, color=(255,255,255))

        print("Ajuste REVERSE_SEGMENT se a direção estiver invertida em algum bloco.")
        print("Deixando todos fracos no final...")
        for seg in range(3):
            white(strip, seg, 40)
        strip.show()

        print("Ctrl+C para sair.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            blackout(strip)
        except Exception:
            pass
        try:
            strip.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
