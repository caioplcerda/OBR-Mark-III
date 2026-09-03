# vision.py corrigido
# Visão mínima para greens com calibração por clique e atualização de config.
# Mais sensível e com logs detalhados.

import cv2
import numpy as np

class Vision:
    def __init__(self, config: dict, log_function=print):
        self.log = log_function if callable(log_function) else print
        self.cfg = {
            "green_h_center": 60,      # centro do H (HSV) para verde
            "green_h_tol": 25,         # tolerância no H
            "green_s_min": 70,
            "green_v_min": 70,
            "min_area": 200,           # menor área mínima do blob verde (era 400)
            "side_margin_frac": 0.10,  # margem lateral para ignorar bordas
        }
        if isinstance(config, dict):
            self.cfg.update(config)

    # ----- Config -----
    def update_config(self, new_cfg: dict):
        if not isinstance(new_cfg, dict):
            return
        self.cfg.update(new_cfg)
        self.log(f"Vision: config atualizado.")

    # ----- Calibração por clique (HSV do pixel clicado) -----
    def calibrate_by_click(self, frame_bgr, x, y, color="green"):
        try:
            # Slice both axes: cvtColor needs a 2D image with 3 channels,
            # (1, 1, 3). frame_bgr[y, x:x+1] collapses the row axis and yields
            # (1, 3), which cvtColor rejects — the exception was caught below,
            # so calibration silently did nothing.
            px_bgr = frame_bgr[int(y):int(y)+1, int(x):int(x)+1]
            if px_bgr.size == 0:
                self.log("Vision: calibrate_by_click out of bounds")
                return False
            px_hsv = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0,0]
            H, S, V = int(px_hsv[0]), int(px_hsv[1]), int(px_hsv[2])
            if color == "green":
                self.cfg["green_h_center"] = H
                self.cfg["green_s_min"] = max(40, min(255, S - 10))
                self.cfg["green_v_min"] = max(40, min(255, V - 10))
                self.log(f"Vision: green calibrado H={H} S>={self.cfg['green_s_min']} V>={self.cfg['green_v_min']}")
                return True
            return False
        except Exception as e:
            self.log(f"Vision: falha calibrate_by_click: {e}")
            return False

    # ----- Green detection -----
    def detect_greens(self, frame_bgr):
        """
        Retorna (centroides, direção)
        - centroides: lista de (x,y) dos blobs verdes
        - direção: 'left', 'right', 'uturn' ou None
        """
        H, W = frame_bgr.shape[:2]
        margin = int(self.cfg["side_margin_frac"] * W)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        h0 = int(self.cfg["green_h_center"])
        tol = int(self.cfg["green_h_tol"])
        s_min = int(self.cfg["green_s_min"])
        v_min = int(self.cfg["green_v_min"])

        # Monte duas faixas (verde pode passar pelo 180°/0° do H)
        lower1 = np.array([max(0, h0 - tol), s_min, v_min], dtype=np.uint8)
        upper1 = np.array([min(179, h0 + tol), 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower1, upper1)

        # Considerar apenas a metade inferior do frame (antes era 40%)
        roi_top = int(H * 0.5)
        mask[:roi_top, :] = 0

        # Ignorar margens laterais
        mask[:, :margin] = 0
        mask[:, W - margin:] = 0

        # Morfologia leve
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        # Contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.cfg["min_area"]:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: 
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            blobs.append(((cx, cy), area))

        blobs.sort(key=lambda x: -x[1])
        cents = [b[0] for b in blobs[:3]]

        direction = None
        if len(cents) >= 2:
            direction = "uturn"
        elif len(cents) == 1:
            cx = cents[0][0]
            direction = "left" if cx < W//2 else "right"

        # Logs detalhados
        if cents:
            self.log(f"Vision: blobs verdes detectados = {len(cents)}, direção = {direction}")
        return cents, direction
