# led_control.py
# 2x módulos de 4 LEDs + 1x módulo de 7 LEDs = 15 LEDs WS2812 (NeoPixel)
# Requisito: 7 LEDs SEMPRE branco forte; 4+4 mostram status com cores mais fracas.
#
# Pino padrão: GPIO12 (PWM, canal 0). Ajuste se precisar.
# Instalação:
#   sudo apt-get install -y python3-pip
#   pip3 install rpi_ws281x adafruit-circuitpython-neopixel
#
# Ligações:
#   DIN -> GPIO12 (pino físico 32), GND comum, 5V dedicado.
#   Recomenda-se resistor ~330Ω em série no DATA e capacitor 1000µF entre 5V e GND.

import time

try:
    from rpi_ws281x import PixelStrip, Color, ws
    RPI_WS281X_OK = True
except Exception:
    RPI_WS281X_OK = False

TOTAL_LEDS = 15

# Índices por módulo
MOD4A = list(range(0, 4))     # primeiro 4 bits
MOD4B = list(range(4, 8))     # segundo 4 bits
MOD7  = list(range(8, 15))    # 7 bits (sempre branco forte)

# Intensidades (não é brilho global; é intensidade por cor)
DIM = {
    "blue":   (0,   0,  80),
    "green":  (0,  90,   0),
    "yellow": (80, 60,   0),
    "amber":  (120, 60,  0),
    "purple": (120, 0, 120),
    "red":    (120, 0,   0),
    "white":  (40, 40,  40),
}

STRONG_WHITE_7 = (255, 255, 255)   # 7 bits sempre forte
DIM_WHITE_4    = DIM["white"]      # base fraca pros 4+4 quando não há status “ativo”


class LedController:
    def __init__(self, pin=12, brightness=128):
        """
        pin: GPIO para o DIN dos WS2812 (GPIO12 recomendado).
        brightness: 0..255 (global do strip).
        """
        self.enabled = False
        self.strip = None

        if not RPI_WS281X_OK:
            print("[LED] rpi_ws281x não disponível — LEDs desabilitados.")
            return

        self.strip = PixelStrip(
            num=TOTAL_LEDS,
            pin=pin,
            freq_hz=800000,
            dma=10,
            invert=False,
            brightness=brightness,            # brilho global
            channel=0,                        # PWM ch0 → GPIO12
            strip_type=ws.WS2811_STRIP_GRB
        )
        self.strip.begin()
        self.enabled = True

        # aplica base inicial: 7 em branco forte; 4+4 em branco fraco
        self._apply_base(show=True)

    # ---------------- helpers ----------------
    @staticmethod
    def _rgb(r, g, b):
        return Color(int(r), int(g), int(b))

    def _set_range(self, indices, color, show=False):
        if not self.enabled: return
        c = self._rgb(*color)
        for i in indices:
            if 0 <= i < self.strip.numPixels():
                self.strip.setPixelColor(i, c)
        if show:
            self.strip.show()

    def _apply_base(self, show=False):
        """Mantém a regra: 7 bits sempre branco forte; 4+4 em branco fraco."""
        if not self.enabled: return
        # 7 bits forte
        self._set_range(MOD7, STRONG_WHITE_7, show=False)
        # 4+4 fraco (será sobrescrito pelos status quando necessário)
        self._set_range(MOD4A + MOD4B, DIM_WHITE_4, show=show)

    def off(self):
        """Apaga tudo (normalmente não usamos, pois queremos os 7 sempre brancos).
        Se chamar off(), reaplique a base com _apply_base()."""
        if not self.enabled: return
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, self._rgb(0, 0, 0))
        self.strip.show()

    # --------------- padrões de status ---------------
    def status_ok(self):
        """OK estável: 4+4 em azul fraco; 7 continua branco forte."""
        if not self.enabled: return
        self._apply_base(show=False)
        self._set_range(MOD4A + MOD4B, DIM["blue"], show=True)

    def status_ahead(self):
        """Interseção à frente (top-first): 4+4 amarelo; 7 mantém branco forte."""
        if not self.enabled: return
        self._apply_base(show=False)
        self._set_range(MOD4A + MOD4B, DIM["yellow"], show=True)

    def status_intersection(self):
        """Interseção confirmada: 4+4 âmbar; 7 mantém branco forte."""
        if not self.enabled: return
        self._apply_base(show=False)
        self._set_range(MOD4A + MOD4B, DIM["amber"], show=True)

    def status_turn(self, direction):
        """
        Verde(s) nas barras de 4 LEDs:
          - left  -> apenas MOD4A verde
          - right -> apenas MOD4B verde
          - uturn -> 4+4 roxo piscando (7 continua branco)
        """
        if not self.enabled: return
        if direction == "left":
            self._apply_base(show=False)
            self._set_range(MOD4A, DIM["green"], show=False)
            self._set_range(MOD4B, DIM_WHITE_4, show=True)
        elif direction == "right":
            self._apply_base(show=False)
            self._set_range(MOD4A, DIM_WHITE_4, show=False)
            self._set_range(MOD4B, DIM["green"], show=True)
        elif direction == "uturn":
            # pisca roxo nas duas barras, mantendo 7 branco
            for _ in range(2):
                self._apply_base(show=False)
                self._set_range(MOD4A + MOD4B, DIM["purple"], show=True)
                time.sleep(0.12)
                self._apply_base(show=True)
                time.sleep(0.12)
        else:
            self.status_ok()

    def status_lost(self):
        """Linha perdida: 4+4 vermelho piscando; 7 continua branco."""
        if not self.enabled: return
        for _ in range(2):
            self._apply_base(show=False)
            self._set_range(MOD4A + MOD4B, DIM["red"], show=True)
            time.sleep(0.08)
            self._apply_base(show=True)
            time.sleep(0.08)

    def status_following(self):
        """Seguindo linha: ‘wipe’ lento azul nas barras de 4; 7 mantém branco."""
        if not self.enabled: return
        self._apply_base(show=False)
        seq = MOD4A + MOD4B
        for i in range(len(seq)):
            # fundo fraco
            self._set_range(seq, DIM_WHITE_4, show=False)
            # pixel atual azul
            idx = seq[i]
            self._set_range([idx], DIM["blue"], show=True)
            time.sleep(0.05)

    def status_ok_idle(self):
        """Mantém só a base (7 branco forte; 4+4 branco fraco)."""
        self._apply_base(show=True)

    def cleanup(self):
        try:
            self._apply_base(show=True)
        except Exception:
            pass
