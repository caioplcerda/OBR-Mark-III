# vision.py
# Visão estilo RCJ 2014: binarização simples, detecção de verdes/vermelhos,
# calibração de cores (HSV) e apoio ao main.py.

import cv2
import numpy as np

class Vision:
    def __init__(self, config=None, log_function=None):
        self.config = config or {}
        self.log = log_function or (lambda m: print(m))

        # Limites HSV básicos (ajustáveis via calibração por clique)
        # preto = linha
        self.LOWER_BLACK = np.array([0, 0, 0])
        self.UPPER_BLACK = np.array([180, 255, 60])

        # branco = fundo
        self.LOWER_WHITE = np.array([0, 0, 180])
        self.UPPER_WHITE = np.array([180, 40, 255])

        # verde = marcadores
        self.LOWER_GREEN = np.array([40, 70, 70])
        self.UPPER_GREEN = np.array([80, 255, 255])

        # vermelho = chegada (duas faixas porque HSV é circular)
        self.LOWER_RED1 = np.array([0, 120, 70])
        self.UPPER_RED1 = np.array([10, 255, 255])
        self.LOWER_RED2 = np.array([170, 120, 70])
        self.UPPER_RED2 = np.array([180, 255, 255])

        self.GREEN_THRESHOLD_AREA = 200  # área mínima pra aceitar verde

    # ==== calibração ====
    def calibrate_by_click(self, frame_bgr, x, y, color):
        """
        Ajusta limites HSV com base em um clique do usuário no stream.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        pixel = hsv[y, x]
        h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])

        tol = 20
        lower = np.array([max(0, h - tol), max(0, s - tol), max(0, v - tol)])
        upper = np.array([min(180, h + tol), min(255, s + tol), min(255, v + tol)])

        if color == "black":
            self.LOWER_BLACK, self.UPPER_BLACK = lower, upper
        elif color == "white":
            self.LOWER_WHITE, self.UPPER_WHITE = lower, upper
        elif color == "green":
            self.LOWER_GREEN, self.UPPER_GREEN = lower, upper
        elif color == "red":
            self.LOWER_RED1, self.UPPER_RED1 = lower, upper
            # cria segunda faixa pra complementar circularidade do vermelho
            self.LOWER_RED2, self.UPPER_RED2 = lower, upper
        else:
            return False

        self.log(f"Calibração {color}: lower={lower}, upper={upper}")
        return True

    # ==== máscaras rápidas (estilo RCJ: tudo em HSV simples) ====
    def mask_black(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.LOWER_BLACK, self.UPPER_BLACK)

    def mask_green(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)

    def mask_red(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.LOWER_RED1, self.UPPER_RED1)
        m2 = cv2.inRange(hsv, self.LOWER_RED2, self.UPPER_RED2)
        return cv2.add(m1, m2)

    # ==== utilitário pra atualizar config salva ====
    def update_config(self, cfg):
        if not cfg:
            return
        if "lower_black" in cfg and "upper_black" in cfg:
            self.LOWER_BLACK = np.array(cfg["lower_black"])
            self.UPPER_BLACK = np.array(cfg["upper_black"])
        if "lower_white" in cfg and "upper_white" in cfg:
            self.LOWER_WHITE = np.array(cfg["lower_white"])
            self.UPPER_WHITE = np.array(cfg["upper_white"])
        if "lower_green" in cfg and "upper_green" in cfg:
            self.LOWER_GREEN = np.array(cfg["lower_green"])
            self.UPPER_GREEN = np.array(cfg["upper_green"])
        if "lower_red1" in cfg and "upper_red1" in cfg:
            self.LOWER_RED1 = np.array(cfg["lower_red1"])
            self.UPPER_RED1 = np.array(cfg["upper_red1"])
        if "lower_red2" in cfg and "upper_red2" in cfg:
            self.LOWER_RED2 = np.array(cfg["lower_red2"])
            self.UPPER_RED2 = np.array(cfg["upper_red2"])
        self.log("Vision config atualizada via update_config")
