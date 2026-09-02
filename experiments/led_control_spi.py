#!/usr/bin/env python3
# led_control_spi.py
# WS2812 via SPI (spidev) — 3 segmentos x 7 LEDs = 21 LEDs (DIN em MOSI /dev/spidev0.0).
# Efeitos só com "branco" (sem mudar cor), usando chase/wipe/blink para estados.
#
# API:
#   leds = LedSPIController(pin=None, brightness=0.6)  # pin ignorado (usamos SPI)
#   leds.status_ok_idle()
#   leds.status_ok()
#   leds.status_ahead()
#   leds.status_intersection()
#   leds.status_lost()
#   leds.status_following()
#   leds.status_turn("left"|"right"|"uturn")
#   leds.cleanup()
#
# Observações:
# - Rode o processo como root ou dê acesso ao /dev/spidev0.0
# - Habilite SPI no sistema (raspi-config ou config.txt).
# - Conexões: MOSI(GPIO10) -> DIN primeiro módulo; 5V e GND comuns.
# - Todos os efeitos são não-bloqueantes: há uma thread de animação com fila.

import time
import threading
import spidev

# ===== Config física / mapeamento =====
LED_COUNT       = 21          # 3 x 7
SEGMENT_SIZE    = 7
SEG_LEFT        = 0
SEG_RIGHT       = 1
SEG_MID         = 2
REVERSE_SEGMENT = [False, False, False]  # se algum módulo estiver invertido fisicamente

# SPI
SPI_BUS    = 0
SPI_DEV    = 0
SPI_MAX_HZ = 3_200_000
SPI_MODE   = 0

# Encoder WS2812 por SPI (nibbles)
NIBBLE_0 = 0b1000
NIBBLE_1 = 0b1110
RESET_TRAILER_LEN = 100  # zeros >80us para latch

# Brilho global (software)
GLOBAL_BRIGHTNESS = 0.60  # 0..1
DIM_WHITE = 40            # nível de "branco fraco" para idle (0..255)
STRONG_WHITE = 220        # nível 'forte' (0..255)

def _clamp8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else x)

def _byte_to_nibbles(b: int) -> int:
    out = 0
    for i in range(8):
        bit = (b & (1 << (7 - i))) != 0
        nib = NIBBLE_1 if bit else NIBBLE_0
        out = (out << 4) | nib
    return out

def _pack32_to_bytes(x: int) -> bytes:
    return bytes([(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF])

def _encode_ws2812_frame(grb_bytes: bytes) -> bytes:
    buf = bytearray()
    for b in grb_bytes:
        buf.extend(_pack32_to_bytes(_byte_to_nibbles(b)))
    buf.extend(b"\x00" * RESET_TRAILER_LEN)
    return bytes(buf)

class _WS2812_SPI:
    def __init__(self, led_count: int):
        self.n = led_count
        self.buf = [(0,0,0)] * self.n  # (G,R,B)
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEV)
        self.spi.max_speed_hz = SPI_MAX_HZ
        self.spi.mode = SPI_MODE

    def set_pixel(self, idx: int, r: int, g: int, b: int, scale: float = GLOBAL_BRIGHTNESS):
        if not (0 <= idx < self.n):
            return
        r = _clamp8(int(r * scale))
        g = _clamp8(int(g * scale))
        b = _clamp8(int(b * scale))
        self.buf[idx] = (g, r, b)  # WS2812 = GRB

    def clear(self):
        for i in range(self.n):
            self.buf[i] = (0, 0, 0)

    def show(self):
        raw = bytearray()
        for (g,r,b) in self.buf:
            raw.extend((g & 0xFF, r & 0xFF, b & 0xFF))
        payload = _encode_ws2812_frame(bytes(raw))
        self.spi.writebytes(payload)

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass

def _seg_bounds(seg_index: int):
    start = seg_index * SEGMENT_SIZE
    end   = start + SEGMENT_SIZE - 1
    return start, end

def _set_segment(strip: _WS2812_SPI, seg_index: int, level: int):
    """Pinta segmento com 'branco' no level (0..255)."""
    start, end = _seg_bounds(seg_index)
    seq = list(range(start, end+1))
    if REVERSE_SEGMENT[seg_index]:
        seq = list(reversed(seq))
    for i in seq:
        strip.set_pixel(i, level, level, level, scale=1.0)

def _blackout(strip: _WS2812_SPI):
    strip.clear()
    strip.show()

class LedSPIController:
    """Controlador com API compatível, usando apenas branco e efeitos de movimento."""
    def __init__(self, pin=None, brightness=STRONG_WHITE):
        # pin ignorado (SPI)
        self.strip = _WS2812_SPI(LED_COUNT)
        self._lock = threading.Lock()
        self._running = True
        self._baseline_dim = DIM_WHITE
        self._queue = []
        self._q_lock = threading.Lock()
        self._anim_th = threading.Thread(target=self._anim_loop, daemon=True)
        self._anim_th.start()
        # estado base: tudo fraco
        self.status_ok_idle()

    # ==== API pública esperada pelo main ====
    @property
    def enabled(self):
        return True

    def status_ok_idle(self):
        """7-branco forte fixo no robô + 3 blocos fracos: base 'normal'."""
        self._enqueue(("idle", {}))

    def status_ok(self):
        """Efeito curto de 'validando' (wipe suave meio->lados)."""
        self._enqueue(("ok", {}))

    def status_ahead(self):
        """AHEAD: chase no bloco do MEIO para frente (duas passadas rápidas)."""
        self._enqueue(("ahead", {}))

    def status_intersection(self):
        """Interseção: wipe 'duplo' indo do meio para os lados simultâneo."""
        self._enqueue(("intersection", {}))

    def status_lost(self):
        """Perdeu a linha: dois blinks rápidos (branco total) e volta pro idle."""
        self._enqueue(("lost", {}))

    def status_following(self):
        """Seguindo: um anel contínuo (esq -> meio -> dir -> meio), discreto."""
        self._enqueue(("following", {}))

    def status_turn(self, direction: str):
        """Curvas: left, right, uturn — chase focado no(s) segmento(s) correspondente(s)."""
        self._enqueue(("turn", {"dir": direction}))

    def cleanup(self):
        self._running = False
        try:
            self._anim_th.join(timeout=0.5)
        except Exception:
            pass
        try:
            _blackout(self.strip)
        except Exception:
            pass
        self.strip.close()

    # ==== motor de animações (thread) ====
    def _enqueue(self, item):
        with self._q_lock:
            # mantém fila curta (evita backlog)
            if len(self._queue) > 8:
                self._queue = self._queue[-8:]
            self._queue.append(item)

    def _pop(self):
        with self._q_lock:
            if self._queue:
                return self._queue.pop(0)
        return None

    def _anim_loop(self):
        # baseline sempre que ocioso
        self._apply_idle()
        while self._running:
            job = self._pop()
            if job is None:
                time.sleep(0.02)
                continue
            kind, data = job
            try:
                if kind == "idle":
                    self._apply_idle()
                elif kind == "ok":
                    self._fx_ok()
                elif kind == "ahead":
                    self._fx_ahead()
                elif kind == "intersection":
                    self._fx_intersection()
                elif kind == "lost":
                    self._fx_lost()
                elif kind == "following":
                    self._fx_following()
                elif kind == "turn":
                    self._fx_turn(data.get("dir"))
                else:
                    self._apply_idle()
            except Exception:
                # em caso de erro, não trava animações; restaura base
                self._apply_idle()

    # ==== efeitos ====
    def _apply_idle(self):
        with self._lock:
            for seg in (SEG_LEFT, SEG_RIGHT, SEG_MID):
                _set_segment(self.strip, seg, self._baseline_dim)
            self.strip.show()

    def _fx_ok(self, steps=3, delay=0.05):
        # "onda" meio -> lados, 3 passos
        with self._lock:
            for _ in range(steps):
                # reforça meio
                _set_segment(self.strip, SEG_MID, STRONG_WHITE)
                self.strip.show(); time.sleep(delay)
                # reforça lados
                _set_segment(self.strip, SEG_LEFT, STRONG_WHITE)
                _set_segment(self.strip, SEG_RIGHT, STRONG_WHITE)
                self.strip.show(); time.sleep(delay)
                # volta base
                self._apply_idle()

    def _fx_ahead(self, loops=2, delay=0.06):
        # chase no segmento do meio
        start, end = _seg_bounds(SEG_MID)
        seq = list(range(start, end+1))
        if REVERSE_SEGMENT[SEG_MID]:
            seq = list(reversed(seq))
        with self._lock:
            for _ in range(loops):
                for i in seq:
                    # base
                    self._apply_idle()
                    # “cabeça” forte
                    self.strip.set_pixel(i, STRONG_WHITE, STRONG_WHITE, STRONG_WHITE, scale=1.0)
                    self.strip.show(); time.sleep(delay)
            self._apply_idle()

    def _fx_intersection(self, loops=2, delay=0.06):
        # wipe duplo: do centro do MEIO para os lados simultâneo em LEFT e RIGHT
        with self._lock:
            for _ in range(loops):
                # passo 1: reforça meio
                _set_segment(self.strip, SEG_MID, STRONG_WHITE); self.strip.show(); time.sleep(delay)
                # passo 2: reforça lados
                _set_segment(self.strip, SEG_LEFT, STRONG_WHITE)
                _set_segment(self.strip, SEG_RIGHT, STRONG_WHITE)
                self.strip.show(); time.sleep(delay)
                self._apply_idle()

    def _fx_lost(self, blinks=2, delay=0.07):
        # dois blinks brancos "full" (sem cor) e retorna
        with self._lock:
            for _ in range(blinks):
                # tudo forte
                for i in range(LED_COUNT):
                    self.strip.set_pixel(i, STRONG_WHITE, STRONG_WHITE, STRONG_WHITE, scale=1.0)
                self.strip.show(); time.sleep(delay)
                # base
                self._apply_idle(); time.sleep(delay)

    def _fx_following(self, loops=1, delay=0.05):
        # "anel" discreto: esquerda -> meio -> direita -> meio
        order = [SEG_LEFT, SEG_MID, SEG_RIGHT, SEG_MID]
        with self._lock:
            for _ in range(loops):
                for seg in order:
                    self._apply_idle()
                    _set_segment(self.strip, seg, STRONG_WHITE)
                    self.strip.show(); time.sleep(delay)
            self._apply_idle()

    def _fx_turn(self, direction: str, loops=2, delay=0.06):
        # left: chase no LEFT; right: chase no RIGHT; uturn: alterna L/R
        if direction == "left":
            seg = SEG_LEFT
            seq = list(range(*_seg_bounds(seg), 1))
        elif direction == "right":
            seg = SEG_RIGHT
            seq = list(range(*_seg_bounds(seg), 1))
        elif direction == "uturn":
            # alterna L/R reforçando blocos
            with self._lock:
                for _ in range(loops):
                    self._apply_idle()
                    _set_segment(self.strip, SEG_LEFT, STRONG_WHITE); self.strip.show(); time.sleep(delay*3/2)
                    self._apply_idle()
                    _set_segment(self.strip, SEG_RIGHT, STRONG_WHITE); self.strip.show(); time.sleep(delay*3/2)
                self._apply_idle()
            return
        else:
            self._apply_idle()
            return

        start, end = _seg_bounds(seg)
        seq = list(range(start, end+1))
        if REVERSE_SEGMENT[seg]:
            seq = list(reversed(seq))
        with self._lock:
            for _ in range(loops):
                for i in seq:
                    self._apply_idle()
                    self.strip.set_pixel(i, STRONG_WHITE, STRONG_WHITE, STRONG_WHITE, scale=1.0)
                    self.strip.show(); time.sleep(delay)
            self._apply_idle()
