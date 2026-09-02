# rcj2014_port.py
# Porta Python do núcleo do RCJ 2014: scanline, scancircle e detecção por derivada.
# Inclui um rastreador simples do "ponto verde" em HSV.

import math
import numpy as np
import cv2

# ======== SCANS ========

def scanline(gray: np.ndarray, center_xy, radius: int):
    """Extrai amostra 1D (linha horizontal) centrada em center_xy (x,y)."""
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x0 = cx - radius
    x1 = cx + radius
    x0c = max(0, x0)
    x1c = min(gray.shape[1], x1)
    line = gray[cy, x0c:x1c]
    if x0c > x0:
        line = np.pad(line, (x0c - x0, 0), mode="edge")
    if x1c < x1:
        line = np.pad(line, (0, x1 - x1c), mode="edge")
    return line.astype(np.int16), x0  # int16 evita overflow na derivada


def scancircle(gray: np.ndarray, center_xy, radius: int, look_angle_deg: float, width_deg: float):
    """Amostra a circunferência e 'janelas' o arco de interesse (look-ahead)."""
    h, w = gray.shape[:2]
    cx, cy = center_xy
    # mesma densidade do C++ (~2πR pontos)
    n = max(8, int(2 * math.pi * radius))
    angs = np.linspace(-math.pi, math.pi, n, endpoint=False)
    xs = np.clip((cx + radius * np.cos(angs)).astype(int), 0, w - 1)
    ys = np.clip((cy + radius * np.sin(angs)).astype(int), 0, h - 1)
    ring = gray[ys, xs].astype(np.int16)

    # mascarar fora da janela (duplicando bordas, como no C++)
    a0 = math.radians(look_angle_deg) + math.pi
    wd = math.radians(width_deg)
    i0 = int((a0 - wd / 2) / (2 * math.pi) * n) % n
    i1 = int((a0 + wd / 2) / (2 * math.pi) * n) % n
    if i0 < i1:
        ring[:i0] = ring[i0]
        ring[i1:] = ring[i1 - 1]
    else:
        # janela cruza o fim do vetor
        ring[i1:i0] = ring[i0]

    return ring, angs


# ======== DETECÇÃO POR DERIVADA ========

def find_line_from_scan(scandata: np.ndarray,
                        index_base,
                        scan_type: str,
                        details: dict,
                        min_line_width: int = 6):
    """
    Localiza a linha pelo par de bordas (máx positivo & máx negativo da derivada) e
    devolve o ponto de interseção no frame.
    - scan_type: "line" (index_base = x_inicial) ou "circle" (index_base = vetor de ângulos)
    """
    # derivada centralizada
    deriv = np.zeros_like(scandata, dtype=np.int16)
    deriv[1:-1] = scandata[:-2] - scandata[2:]

    # picos de borda
    left_idx = int(np.argmax(deriv))        # maior positivo
    right_idx = int(np.argmin(deriv))       # mais negativo
    if left_idx > right_idx:
        left_idx, right_idx = right_idx, left_idx

    if right_idx - left_idx < max(3, min_line_width):
        return None, deriv

    if scan_type == "line":
        x0 = int(index_base + (left_idx + right_idx) // 2)
        y0 = int(details["center_point"][1])
        return (x0, y0), deriv

    elif scan_type == "circle":
        angs = index_base
        cx, cy = details["center_point"]
        r = details["radius"]
        pos = (left_idx + right_idx) // 2
        ang = float(angs[pos])
        x0 = int(cx + r * math.cos(ang))
        y0 = int(cy + r * math.sin(ang))
        return (x0, y0), deriv

    return None, deriv


def line_angle_from_points(p1, p2):
    if not p1 or not p2:
        return 0.0
    # no RCJ o ângulo é referente à direção da linha no plano da imagem
    return math.degrees(math.atan2(p2[0] - p1[0], -(p2[1] - p1[1])))


# ======== RASTREAMENTO DO “VERDE” (HSV) ========

def track_green_centroids(bgr_frame: np.ndarray, hsv_bounds, area_min: int = 100):
    """Retorna centróides dos blobs verdes e classifica direção (left/right/uturn/straight/None)."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_bounds["lower"], hsv_bounds["upper"])
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    cx_mid = w // 2

    centroids = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        gx = int(M["m10"] / M["m00"])
        gy = int(M["m01"] / M["m00"])
        centroids.append((gx, gy))

    direction = None
    if centroids:
        left = any(gx < cx_mid - 50 for gx, _ in centroids)
        right = any(gx > cx_mid + 50 for gx, _ in centroids)
        if left and right:
            direction = "uturn"
        elif left:
            direction = "left"
        elif right:
            direction = "right"
        else:
            direction = "straight"

    return centroids, direction, mask
