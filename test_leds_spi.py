# test_leds_spi.py
# WS2812/NeoPixel via SPI puro (GPIO10 MOSI) no Raspberry Pi (inclui Pi 5).
# Não usa rpi_ws281x. Requer: sudo apt-get install python3-spidev
# Ligue DIN -> GPIO10 (MOSI / pino 19), GND comum, +5V nos LEDs.

import spidev
import time

# ----- CONFIG DO SEU ARRANJO -----
N_PIXELS = 15             # 7 + 4 + 4
ORDER = "GRB"             # maioria dos módulos é GRB; mude p/ "RGB" se cores saírem erradas
SPI_BUS = 0
SPI_DEV = 0
SPI_HZ = 2_400_000        # 2.4 MHz (cada bit WS2812 vira 3 bits SPI: '0'->100, '1'->110)
RESET_US = 80             # >50us para reset do WS2812

# Mapas: [0..6] = módulo 7 bits; [7..10] e [11..14] = módulos 4 bits
SEG_7 = range(0, 7)
SEG_4A = range(7, 14)
SEG_4B = range(14, 18)

# Intensidades
WHITE_STRONG = (255, 255, 255)
WHITE_WEAK   = (60, 60, 60)

# ----- ENCODER SPI (0->100, 1->110) LUT de 256 bytes -> 3 bytes -----
def _build_lut(order="GRB"):
    # para cada byte (0..255), gera 24 bits SPI (3 bytes) usando '1'->110, '0'->100
    lut = [None] * 256
    for b in range(256):
        bits = []
        for i in range(7, -1, -1):
            if (b >> i) & 1:
                bits += [1,1,0]
            else:
                bits += [1,0,0]
        # empacota 24 bits em 3 bytes
        out = [0,0,0]
        for i, bit in enumerate(bits):
            out[i//8] = (out[i//8] << 1) | bit
        lut[b] = bytes(out)
    return lut

LUT = _build_lut(ORDER)

def color_bytes(r, g, b):
    if ORDER == "GRB":
        return (g, r, b)
    elif ORDER == "RGB":
        return (r, g, b)
    else:
        raise ValueError("ORDER inválido (use 'GRB' ou 'RGB').")

class WS2812_SPI:
    def __init__(self, n, bus=0, dev=0, hz=2_400_000, order="GRB"):
        self.n = n
        self.order = order
        self.spi = spidev.SpiDev()
        self.spi.open(bus, dev)
        self.spi.max_speed_hz = hz
        self.spi.mode = 0
        self.buf = [(0,0,0)] * n

    def set_pixel(self, i, rgb):
        if 0 <= i < self.n:
            self.buf[i] = rgb

    def fill(self, rgb):
        self.buf = [rgb] * self.n

    def show(self):
        # Converte todos os pixels para fluxo SPI
        out = bytearray()
        for (r, g, b) in self.buf:
            grb = color_bytes(r, g, b)
            out += LUT[grb[0]] + LUT[grb[1]] + LUT[grb[2]]
        self.spi.xfer2(out)
        # Reset >50us
        time.sleep(RESET_US / 1_000_000.0)

    def close(self):
        try:
            # apaga tudo
            self.fill((0,0,0))
            self.show()
        finally:
            self.spi.close()


def main():
    strip = WS2812_SPI(N_PIXELS, SPI_BUS, SPI_DEV, SPI_HZ, ORDER)
    try:
        # 7 primeiros: branco forte
        for i in SEG_4A:
            strip.set_pixel(i, WHITE_STRONG)
        # 4+4: branco fraco
        for i in SEG_7:
            strip.set_pixel(i, WHITE_WEAK)
        for i in SEG_4B:
            strip.set_pixel(i, WHITE_WEAK)
        strip.show()
        print("Mostrando: 7 fortes + 4+4 fracos (fixo) por 3s...")
        time.sleep(3)

        # Pequeno ciclo de status nos 4+4
        print("Ciclo de status nos 4+4...")
        def set_status(color):
            for i in SEG_7:
                strip.set_pixel(i, color)
            for i in SEG_4B:
                strip.set_pixel(i, color)
            strip.show()

        # OK (azul fraco)
        set_status((0, 0, 80)); time.sleep(1)
        # AHEAD (ciano fraco)
        set_status((0, 60, 60)); time.sleep(1)
        # INTERSECTION (amarelo fraco)
        set_status((60, 60, 0)); time.sleep(1)
        # LOST (vermelho fraco)
        set_status((80, 0, 0)); time.sleep(1)
        # FOLLOW (verde fraco)
        set_status((0, 80, 0)); time.sleep(1)
        # TURN LEFT (magenta fraco)
        set_status((60, 0, 60)); time.sleep(1)
        # TURN RIGHT (laranja fraco)
        set_status((100, 50, 0)); time.sleep(1)

        # volta ao padrão: 7 fortes + 4+4 fracos
        for i in SEG_4A:
            strip.set_pixel(i, WHITE_STRONG)
        for i in SEG_7:
            strip.set_pixel(i, WHITE_WEAK)
        for i in SEG_4B:
            strip.set_pixel(i, WHITE_WEAK)
        strip.show()
        print("Fim do teste. Ctrl+C para sair, apaga em seguida.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        strip.close()
        print("LEDs apagados e SPI fechado.")


if __name__ == "__main__":
    main()
