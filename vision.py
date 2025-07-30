import cv2
import numpy as np

class Vision:
    """ Classe para todo o processamento de visão computacional. """
    def __init__(self, config, log_function):
        self.config = config
        self.log = log_function
        # === Limites de cor HSV para detecção ===
        self.LOWER_GREEN = np.array([40, 50, 50])
        self.UPPER_GREEN = np.array([85, 255, 255])

        # Limites para a cor Prata/Cinza
        self.LOWER_SILVER = np.array([0, 0, 100])
        self.UPPER_SILVER = np.array([180, 30, 220])

        # Limites para a cor Vermelha (duas faixas no HSV)
        self.LOWER_RED1 = np.array([0, 70, 50])
        self.UPPER_RED1 = np.array([10, 255, 255])
        self.LOWER_RED2 = np.array([170, 70, 50])
        self.UPPER_RED2 = np.array([180, 255, 255])

        # === Parâmetros de Visão ===
        self.FRAME_WIDTH = 640
        self.CENTER_X = self.FRAME_WIDTH // 2
        self.GREEN_THRESHOLD_AREA = 5000
        self.OBSTACLE_MIN_AREA = 2000
        self.OBSTACLE_REGION_Y = 100

    def detect_line_features(self, frame):
        """ Detecta características da linha, como centroide, interseções e obstáculos. """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_black = self.config['hsv_black']['lower']
        upper_black = self.config['hsv_black']['upper']
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        mask_green = cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)

        # Múltiplos ROIs (Regiões de Interesse) para uma análise mais robusta da linha
        rois = {
            "bottom": mask_black[220:240, :],
            "middle": mask_black[180:200, :],
            "top": mask_black[140:160, :]
        }

        centroids = {}
        for name, roi in rois.items():
            M = cv2.moments(roi)
            if M["m00"] != 0:
                centroids[name] = int(M["m10"] / M["m00"])
            else:
                centroids[name] = -1

        cx = centroids["bottom"]  # O centroide principal é o mais próximo do robô

        # Uma interseção é detectada se a linha estiver presente em todos os ROIs
        intersection = all(c != -1 for c in centroids.values())

        # Detecção de cores em área ampla
        mask_silver = cv2.inRange(hsv, self.LOWER_SILVER, self.UPPER_SILVER)
        mask_red1 = cv2.inRange(hsv, self.LOWER_RED1, self.UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, self.LOWER_RED2, self.UPPER_RED2)
        mask_red = cv2.add(mask_red1, mask_red2)

        silver_detected = cv2.countNonZero(mask_silver) > self.GREEN_THRESHOLD_AREA # Reutilizando o threshold
        red_detected = cv2.countNonZero(mask_red) > self.GREEN_THRESHOLD_AREA

        obstacle = self.detect_obstacle(mask_black)

        # Calcula a curvatura
        curvature = 0
        if centroids['top'] != -1 and centroids['bottom'] != -1:
            curvature = centroids['top'] - centroids['bottom']

        # Contagem de pixels na máscara preta
        pixel_count = cv2.countNonZero(mask_black)

        return cx, silver_detected, red_detected, obstacle, intersection, centroids, curvature, mask_black, pixel_count

    def calibrate_by_click(self, frame, x, y, color_name):
        """ Calibra a faixa HSV de uma cor com base em um pixel clicado. """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel_hsv = hsv_frame[y, x]

        h, s, v = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

        # Define uma tolerância para criar a faixa
        h_tolerance = 10
        s_tolerance = 40
        v_tolerance = 40

        lower_bound = np.array([max(0, h - h_tolerance), max(0, s - s_tolerance), max(0, v - v_tolerance)])
        upper_bound = np.array([min(180, h + h_tolerance), min(255, s + s_tolerance), min(255, v + v_tolerance)])

        # Atualiza a configuração global
        if color_name == 'black':
            self.config['hsv_black']['lower'] = lower_bound
            self.config['hsv_black']['upper'] = upper_bound
            self.log(f"Nova calibração para PRETO: {lower_bound} a {upper_bound}")

        # Adicionar lógica para outras cores (verde, etc.) aqui se necessário

    def detect_obstacle(self, mask_black):
        """ Detecta obstáculos na pista. """
        obstacle_roi = mask_black[self.OBSTACLE_REGION_Y - 10:self.OBSTACLE_REGION_Y + 10, self.CENTER_X - 40:self.CENTER_X + 40]
        contours, _ = cv2.findContours(obstacle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.OBSTACLE_MIN_AREA:
                return True
        return False

    def detect_balls(self, frame):
        """ Detecta as bolas de resgate (prateadas e pretas). """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_silver = cv2.inRange(hsv, self.LOWER_SILVER, self.UPPER_SILVER)
        mask_black = cv2.inRange(hsv, self.LOWER_BLACK, self.UPPER_BLACK)

        balls = []

        contours_silver, _ = cv2.findContours(mask_silver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_silver:
            if cv2.contourArea(c) > 200:
                x, y, w, h = cv2.boundingRect(c)
                balls.append({"tipo": "prata", "pos": (x + w // 2, y + h // 2)})

        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_black:
            if cv2.contourArea(c) > 200:
                x, y, w, h = cv2.boundingRect(c)
                balls.append({"tipo": "preta", "pos": (x + w // 2, y + h // 2)})

        return balls
