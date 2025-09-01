# vision.py
# HSV/Greens: calibração por clique + detecção robusta (ROI/forma) dos marcadores.

import cv2
import numpy as np

class Vision:
    def __init__(self, config: dict = None, log_function=print):
        self.log = log_function
        self.config = config or {}

        self.WIDTH = self.config.get("width", 640)
        self.HEIGHT = self.config.get("height", 480)
        self.CENTER_X = self.WIDTH // 2

        # HSV defaults (ajuste via clique na UI)
        self.LOWER_GREEN = np.array(self.config.get("lower_green", [40, 60, 50]), dtype=np.uint8)
        self.UPPER_GREEN = np.array(self.config.get("upper_green", [85, 255, 255]), dtype=np.uint8)

        self.LOWER_RED1 = np.array(self.config.get("lower_red1", [0, 120, 60]), dtype=np.uint8)
        self.UPPER_RED1 = np.array(self.config.get("upper_red1", [10, 255, 255]), dtype=np.uint8)
        self.LOWER_RED2 = np.array(self.config.get("lower_red2", [170, 120, 60]), dtype=np.uint8)
        self.UPPER_RED2 = np.array(self.config.get("upper_red2", [180, 255, 255]), dtype=np.uint8)

        self.GREEN_THRESHOLD_AREA = int(self.config.get("green_min_area", 150))
        self.GREEN_Y1 = int(self.config.get("green_y1", 120))
        self.GREEN_Y2 = int(self.config.get("green_y2", 220))

        self._last_click_hsv = None

    def update_config(self, cfg: dict):
        if not cfg:
            return
        self.config.update(cfg)
        self.WIDTH = self.config.get("width", self.WIDTH)
        self.HEIGHT = self.config.get("height", self.HEIGHT)
        self.CENTER_X = self.WIDTH // 2

        if "lower_green" in self.config: self.LOWER_GREEN = np.array(self.config["lower_green"], dtype=np.uint8)
        if "upper_green" in self.config: self.UPPER_GREEN = np.array(self.config["upper_green"], dtype=np.uint8)

        if "lower_red1" in self.config: self.LOWER_RED1 = np.array(self.config["lower_red1"], dtype=np.uint8)
        if "upper_red1" in self.config: self.UPPER_RED1 = np.array(self.config["upper_red1"], dtype=np.uint8)
        if "lower_red2" in self.config: self.LOWER_RED2 = np.array(self.config["lower_red2"], dtype=np.uint8)
        if "upper_red2" in self.config: self.UPPER_RED2 = np.array(self.config["upper_red2"], dtype=np.uint8)

        self.GREEN_THRESHOLD_AREA = int(self.config.get("green_min_area", self.GREEN_THRESHOLD_AREA))
        self.GREEN_Y1 = int(self.config.get("green_y1", self.GREEN_Y1))
        self.GREEN_Y2 = int(self.config.get("green_y2", self.GREEN_Y2))

    # ===== calibração por clique =====
    def calibrate_by_click(self, frame_bgr, x, y, color: str):
        if frame_bgr is None:
            return False
        x = int(np.clip(x, 0, frame_bgr.shape[1]-1))
        y = int(np.clip(y, 0, frame_bgr.shape[0]-1))
        roi = frame_bgr[max(0, y-3):min(self.HEIGHT, y+4), max(0, x-3):min(self.WIDTH, x+4)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_min, s_min, v_min = int(np.percentile(h, 10)), int(np.percentile(s, 10)), int(np.percentile(v, 10))
        h_max, s_max, v_max = int(np.percentile(h, 90)), int(np.percentile(s, 90)), int(np.percentile(v, 90))

        if color.lower() == "green":
            self.LOWER_GREEN = np.array([max(0, h_min-5), max(0, s_min-15), max(0, v_min-15)], dtype=np.uint8)
            self.UPPER_GREEN = np.array([min(180, h_max+5), min(255, s_max+15), min(255, v_max+15)], dtype=np.uint8)
            self._last_click_hsv = ("green", (self.LOWER_GREEN.tolist(), self.UPPER_GREEN.tolist()))
            self.log(f"HSV GREEN ajustado: {self.LOWER_GREEN.tolist()} .. {self.UPPER_GREEN.tolist()}")
            return True
        elif color.lower() == "red":
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
            self.log("HSV RED ajustado.")
            return True
        else:
            return False

    # ===== verdes (ROI + forma) =====
    def detect_greens(self, frame_bgr):
        """Retorna (centroides, direção: left/right/uturn/None)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)

        y1, y2 = int(self.GREEN_Y1), int(self.GREEN_Y2)
        y1 = max(0, min(self.HEIGHT-1, y1))
        y2 = max(y1+1, min(self.HEIGHT, y2))
        roi = mask[y1:y2, :]

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.GREEN_THRESHOLD_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area == 0:
                continue
            rectangularity = float(area) / rect_area
            aspect = w / float(h)
            if rectangularity < 0.6 or not (0.7 <= aspect <= 1.3):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            gx = int(M["m10"] / M["m00"])
            gy = int(M["m01"] / M["m00"]) + y1
            centroids.append((gx, gy))

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
