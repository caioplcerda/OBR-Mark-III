#!/usr/bin/env python3
# test_leds_spi.py
# Três módulos WS2812 (7 LEDs cada) em série (total 21) usando SPI (spidev).
# Mapeamento:
#   segmento 0: "esquerda"  -> pixels  0..6
#   segmento 1: "esquerda2" -> pixels  7..13
#   segmento 2: "meio"      -> pixels 14..20
#
# Se algum bloco acende invertido, marque REVERSE_SEGMENT[seg] = True.
#
# Execução:
#   sudo python3 test_leds_spi.py
#
# Notas:
# - Usa codificação 1-wire do WS2812 por SPI com nibbles 0/1 -> 0b1000/0b1110 (timming compatível).
# - Clock SPI recomendado: ~3.2 MHz (ajustável abaixo).
# - Reset/latch é obtido enviando um "trailer" de zeros > 80 µs.

import time
import sys
import spidev

# ========= CONFIGURAÇÃO GERAL =========
LED_COUNT        = 21                 # 3 x 7
SEGMENT_SIZE     = 7
SEGMENT_ORDER    = ["esquerda", "esquerda2", "meio"]  # rótulos (só informativo)
REVERSE_SEGMENT  = [False, False, False]              # inverte ordem do segmento se True

SPI_BUS          = 0
SPI_DEV          = 0
SPI_MAX_HZ       = 3_200_000          # ~3.2 MHz funciona bem na maioria dos Pi
SPI_MODE         = 0

# Brilho "global" simples (0.0..1.0) aplicado no software (não é como APA102)
GLOBAL_BRIGHTNESS = 0.6

# Trailer para reset/latch do WS2812 (zeros suficientes > 80us).
# A ~3.2MHz, 100 bytes de zero dão ~250us, folgado.
RESET_TRAILER_LEN = 100

# ========= ENCODER WS2812 por SPI =========
# Codificação por nibble (4 bits) para representar cada bit do WS2812:
#   bit '0' -> 0b1000 (pulso curto)
#   bit '1' -> 0b1110 (pulso longo)
# Cada byte (8 bits) vira 8 nibbles (= 32 bits), que empacotamos em 4 bytes.
# Portanto, para N LEDs (24 bytes RGB), o frame SPI tem 4x mais bytes + trailer.

NIBBLE_0 = 0b1000
NIBBLE_1 = 0b1110

def _byte_to_nibbles(b: int) -> int:
    """Converte 8 bits (MSB->LSB) em 32 bits (8 nibbles) embalados num int de 32 bits."""
    out = 0
    for i in range(8):
        bit = (b & (1 << (7 - i))) != 0
        nib = NIBBLE_1 if bit else NIBBLE_0
        out = (out << 4) | nib
    return out

def _pack32_to_bytes(x: int) -> bytes:
    """Empacota um inteiro de 32 bits em 4 bytes (big-endian)."""
    return bytes([(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF])

def encode_ws2812_frame(rgb_bytes: bytes) -> bytes:
    """
    Converte uma sequência de bytes GRB/WS2812 para o fluxo SPI codificado.
    Esperado: ordem WS2812 = G, R, B (por LED).
    """
    buf = bytearray()
    for b in rgb_bytes:
        nib32 = _byte_to_nibbles(b)
        buf.extend(_pack32_to_bytes(nib32))
    # trailer de zeros para reset/latch
    buf.extend(b"\x00" * RESET_TRAILER_LEN)
    return bytes(buf)

# ========= FRAME BUFFER =========
# WS2812 espera ordem G, R, B por LED.
def clamp8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else x)

class WS2812_SPI:
    def __init__(self, led_count: int):
        self.n = led_count
        # buffer lógico em GRB
        self.buf = [(0, 0, 0)] * self.n  # (G, R, B)
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEV)
        self.spi.max_speed_hz = SPI_MAX_HZ
        self.spi.mode = SPI_MODE
        # Nota: spidev usa bytes-like; vamos enviar com writebytes/ xfer2.

    def set_pixel(self, idx: int, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
        if not (0 <= idx < self.n):
            return
        if scale is None:
            scale = 1.0
        r = clamp8(int(r * scale))
        g = clamp8(int(g * scale))
        b = clamp8(int(b * scale))
        # guarda em GRB
        self.buf[idx] = (g, r, b)

    def fill(self, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
        for i in range(self.n):
            self.set_pixel(i, r, g, b, scale)

    def clear(self):
        self.fill(0, 0, 0, scale=1.0)

    def show(self):
        # Achata GRB em bytes
        raw = bytearray()
        for (g, r, b) in self.buf:
            raw.extend((g & 0xFF, r & 0xFF, b & 0xFF))
        payload = encode_ws2812_frame(bytes(raw))
        # Envia; writebytes é adequado para buffers grandes
        self.spi.writebytes(payload)

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass

# ========= SEGMENTOS =========
def seg_bounds(seg_index: int):
    """Faixa de 7 LEDs para o segmento lógico (0..2)."""
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
    # esquerda2 = verde
    set_segment(strip, 1, 0, 255, 0, scale=1.0)
    # meio = azul
    set_segment(strip, 2, 0, 0, 255, scale=1.0)
    strip.show()
    print("Identificação: esquerda=VERMELHO, esquerda2=VERDE, meio=AZUL")
    time.sleep(2.0)

def demo_chase(strip: WS2812_SPI, seg_index: int, loops=2, delay=0.07, color=(255,255,255)):
    r, g, b = color
    start, end = seg_bounds(seg_index)
    seq = list(range(start, end + 1))
    if REVERSE_SEGMENT[seg_index]:
        seq = list(reversed(seq))
    for _ in range(loops):
        for i in range(len(seq)):
            # apaga segmento
            for j in seq:
                strip.set_pixel(j, 0, 0, 0, scale=1.0)
            # acende cabeça
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

        # Passo 1: identificar blocos com cores
        demo_identify(strip)

        # Passo 2: todos fracos (status base)
        for seg in range(3):
            white(strip, seg, 40)   # fraco
        strip.show()
        time.sleep(1.2)

        # Passo 3: “forte” por segmento para conferir mapeamento
        for seg, name in enumerate(SEGMENT_ORDER):
            print(f"Fortalecendo {name}...")
            white(strip, seg, 220)   # forte
            strip.show()
            time.sleep(1.2)
            white(strip, seg, 40)    # volta fraco
            strip.show()
            time.sleep(0.3)

        # Passo 4: chase para ver orientação física
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
