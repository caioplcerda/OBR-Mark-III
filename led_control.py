# led_control.py
# Simple controller for a single strip of WS2812 (NeoPixel) LEDs.

try:
    from rpi_ws281x import PixelStrip, Color, ws
    RPI_WS281X_OK = True
except Exception:
    RPI_WS281X_OK = False

TOTAL_LEDS = 21

class LedController:
    def __init__(self, pin=12, brightness=128):
        """
        pin: GPIO for the DIN of the WS2812 LEDs (GPIO12 recommended).
        brightness: 0..255 (global brightness for the strip).
        """
        self.enabled = False
        self.strip = None

        if not RPI_WS281X_OK:
            print("[LED] rpi_ws281x library not available — LEDs disabled.")
            return

        try:
            self.strip = PixelStrip(
                num=TOTAL_LEDS,
                pin=pin,
                freq_hz=800000,
                dma=10,
                invert=False,
                brightness=brightness,
                channel=0,
                strip_type=ws.WS2811_STRIP_GRB
            )
            self.strip.begin()
            self.enabled = True
            print(f"[LED] LedController initialized for {TOTAL_LEDS} LEDs.")
        except Exception as e:
            print(f"[LED] Failed to initialize PixelStrip: {e}")
            self.enabled = False

    @staticmethod
    def _rgb(r, g, b):
        """Converts RGB tuple to Color object."""
        return Color(int(r), int(g), int(b))

    def set_all(self, r, g, b, show=True):
        """
        Sets all LEDs in the strip to the same color.
        """
        if not self.enabled:
            return

        color_obj = self._rgb(r, g, b)
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, color_obj)

        if show:
            self.strip.show()

    def cleanup(self):
        """Turn off all LEDs on cleanup."""
        if not self.enabled:
            return
        self.set_all(0, 0, 0)
