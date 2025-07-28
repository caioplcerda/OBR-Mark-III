import cv2
import numpy as np

class Vision:
    """ Classe para todo o processamento de visão computacional. """
    def __init__(self, config):
        self.config = config
        # === Limites de cor HSV para detecção ===
        self.LOWER_GREEN = np.array([40, 50, 50])
        self.UPPER_GREEN = np.array([85, 255, 255])
        self.LOWER_SILVER = np.array([0, 0, 180])
        self.UPPER_SILVER = np.array([180, 50, 255])

        # === Parâmetros de Visão ===
        self.FRAME_WIDTH = 640
        self.CENTER_X = self.FRAME_WIDTH // 2
        self.GREEN_THRESHOLD_AREA = 5000
        self.OBSTACLE_MIN_AREA = 2000
        self.OBSTACLE_REGION_Y = 100

    def detect_line_features(self, frame):
        """ Detecta características da linha, como centroide, interseções e obstáculos. """
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

        green_detected = cv2.countNonZero(mask_green) > self.GREEN_THRESHOLD_AREA
        obstacle = self.detect_obstacle(mask_black)

        return cx, green_detected, obstacle, intersection, mask_black, mask_green, centroids

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
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
