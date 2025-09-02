# vision.py
# HSV/Greens: calibração por clique + detecção robusta (ROI/forma) dos marcadores.
# Compatível com main.py:
#   - Vision({}, log)
#   - vision.update_config(cfg)
#   - vision.calibrate_by_click(frame, x, y, color)
#   - centroids, direction = vision.detect_greens(frame)

import cv2
import numpy as np

class Vision:
    def __init__(self, config: dict = None, log_function=print):
        self.log = log_function
        self.config = (config or {}).copy()

        # Dimensões default (atualizadas por update_config se vierem do config.json)
        self.WIDTH = int(self.config.get("width", 640))
        self.HEIGHT = int(self.config.get("height", 480))
        self.CENTER_X = self.WIDTH // 2

        # Faixa HSV padrão (ajuste via clique)
        self.LOWER_GREEN = np.array(self.config.get("lower_green", [40, 60, 50]), dtype=np.uint8)
        self.UPPER_GREEN = np.array(self.config.get("upper_green", [85, 255, 255]), dtype=np.uint8)

        # Vermelho opcional (se um dia quiser usar)
        self.LOWER_RED1 = np.array(self.config.get("lower_red1", [0, 120, 60]), dtype=np.uint8)
        self.UPPER_RED1 = np.array(self.config.get("upper_red1", [10, 255, 255]), dtype=np.uint8)
        self.LOWER_RED2 = np.array(self.config.get("lower_red2", [170, 120, 60]), dtype=np.uint8)
        self.UPPER_RED2 = np.array(self.config.get("upper_red2", [180, 255, 255]), dtype=np.uint8)

        # Parâmetros de detecção dos verdes
        self.GREEN_THRESHOLD_AREA = int(self.config.get("green_min_area", 150))  # área mínima do contorno
        # ROI vertical onde os verdes aparecem (ajuste fino se necessário)
        self.GREEN_Y1 = int(self.config.get("green_y1", 120))
        self.GREEN_Y2 = int(self.config.get("green_y2", 220))

        self._last_click_hsv = None

    # =========================================================
    # Atualiza parâmetros via config.json / UI
    # =========================================================
    def update_config(self, cfg: dict):
        if not cfg:
            return
        self.config.update(cfg)

        if "width" in cfg or "height" in cfg:
            self.WIDTH = int(self.config.get("width", self.WIDTH))
            self.HEIGHT = int(self.config.get("height", self.HEIGHT))
            self.CENTER_X = self.WIDTH // 2

        if "lower_green" in cfg:
            self.LOWER_GREEN = np.array(self.config["lower_green"], dtype=np.uint8)
        if "upper_green" in cfg:
            self.UPPER_GREEN = np.array(self.config["upper_green"], dtype=np.uint8)

        if "lower_red1" in cfg:
            self.LOWER_RED1 = np.array(self.config["lower_red1"], dtype=np.uint8)
        if "upper_red1" in cfg:
            self.UPPER_RED1 = np.array(self.config["upper_red1"], dtype=np.uint8)
        if "lower_red2" in cfg:
            self.LOWER_RED2 = np.array(self.config["lower_red2"], dtype=np.uint8)
        if "upper_red2" in cfg:
            self.UPPER_RED2 = np.array(self.config["upper_red2"], dtype=np.uint8)

        if "green_min_area" in cfg:
            self.GREEN_THRESHOLD_AREA = int(self.config["green_min_area"])
        if "green_y1" in cfg:
            self.GREEN_Y1 = int(self.config["green_y1"])
        if "green_y2" in cfg:
            self.GREEN_Y2 = int(self.config["green_y2"])

    # =========================================================
    # Calibração por clique no frame
    # =========================================================
    def calibrate_by_click(self, frame_bgr, x, y, color: str):
        """Ajusta faixa HSV do 'green' (ou 'red') a partir de um clique no preview."""
        if frame_bgr is None:
            return False

        x = int(np.clip(x, 0, frame_bgr.shape[1]-1))
        y = int(np.clip(y, 0, frame_bgr.shape[0]-1))

        # pequena janela em torno do clique
        x0 = max(0, x - 3); x1 = min(frame_bgr.shape[1], x + 4)
        y0 = max(0, y - 3); y1 = min(frame_bgr.shape[0], y + 4)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_min, s_min, v_min = int(np.percentile(h, 10)), int(np.percentile(s, 10)), int(np.percentile(v, 10))
        h_max, s_max, v_max = int(np.percentile(h, 90)), int(np.percentile(s, 90)), int(np.percentile(v, 90))

        color = (color or "").strip().lower()
        if color == "green":
            self.LOWER_GREEN = np.array([max(0, h_min-5), max(0, s_min-15), max(0, v_min-15)], dtype=np.uint8)
            self.UPPER_GREEN = np.array([min(180, h_max+5), min(255, s_max+15), min(255, v_max+15)], dtype=np.uint8)
            self._last_click_hsv = ("green", (self.LOWER_GREEN.tolist(), self.UPPER_GREEN.tolist()))
            self.log(f"[Vision] HSV GREEN -> {self.LOWER_GREEN.tolist()} .. {self.UPPER_GREEN.tolist()}")
            return True

        if color == "red":
            # trata os dois intervalos do vermelho (wrap no H)
            base_low = np.array([h_min, s_min, v_min], dtype=np.int32)
            base_up  = np.array([h_max, s_max, v_max], dtype=np.int32)
            if (h_min + h_max)/2 > 150:
                self.LOWER_RED2 = np.clip(base_low - np.array([5,15,15]), 0, [180,255,255]).astype(np.uint8)
                self.UPPER_RED2 = np.clip(base_up  + np.array([5,15,15]), 0, [180,255,255]).astype(np.uint8)
            else:
                self.LOWER_RED1 = np.clip(base_low - np.array([5,15,15]), 0, [180,255,255]).astype(np.uint8)
                self.UPPER_RED1 = np.clip(base_up  + np.array([5,15,15]), 0, [180,255,255]).astype(np.uint8)
            self._last_click_hsv = ("red", (
                self.LOWER_RED1.tolist(), self.UPPER_RED1.tolist(),
                self.LOWER_RED2.tolist(), self.UPPER_RED2.tolist()
            ))
            self.log("[Vision] HSV RED ajustado.")
            return True

        return False

    # =========================================================
    # Detecção de marcadores verdes (centroides + direção)
    # =========================================================
    def detect_greens(self, frame_bgr):
        """
        Retorna (centroides, direção):
          - centroides: lista [(gx, gy), ...] dentro da ROI vertical
          - direção: "left", "right", "uturn" ou None
        """
        # Converte para HSV e aplica máscara dos verdes
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)

        # ROI vertical onde esperamos ver os verdes
        y1, y2 = int(self.GREEN_Y1), int(self.GREEN_Y2)
        y1 = max(0, min(self.HEIGHT-1, y1))
        y2 = max(y1+1, min(self.HEIGHT, y2))
        roi = mask[y1:y2, :]

        # Contornos
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.GREEN_THRESHOLD_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area <= 0:
                continue
            rectangularity = float(area) / rect_area
            aspect = w / float(h) if h > 0 else 0
            # filtros simples de forma (quase quadrado)
            if rectangularity < 0.6 or not (0.7 <= aspect <= 1.3):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            gx = int(M["m10"] / M["m00"])
            gy = int(M["m01"] / M["m00"]) + y1
            centroids.append((gx, gy))

        # Direção com histerese simples (left/right/uturn)
        direction = None
        if centroids:
            left  = any(gx < (self.CENTER_X - 50) for gx, _ in centroids)
            right = any(gx > (self.CENTER_X + 50) for gx, _ in centroids)
            if left and right:
                direction = "uturn"
            elif left:
                direction = "left"
            elif right:
                direction = "right"

        return centroids, direction
