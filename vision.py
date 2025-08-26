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
        kernel = np.ones((5, 5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)

        mask_green = cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)

        # Múltiplos ROIs (Regiões de Interesse) para uma análise mais robusta da linha
        rois = {
            "bottom": mask_black[210:240, :],
            "middle": mask_black[160:200, :],
            "top": mask_black[110:150, :]
        }

        centroids = {}
        for name, roi in rois.items():
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    centroids[name] = int(M["m10"] / M["m00"])
                else:
                    centroids[name] = -1
            else:
                centroids[name] = -1

        cx = centroids["bottom"]  # O centroide principal é o mais próximo do robô

        # Interseção detectada se a linha aparecer em todos os ROIs
        intersection = all(c != -1 for c in centroids.values())

        # Detecção da linha de chegada (vermelho)
        mask_red1 = cv2.inRange(hsv, self.LOWER_RED1, self.UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, self.LOWER_RED2, self.UPPER_RED2)
        mask_red = cv2.add(mask_red1, mask_red2)
        red_detected = cv2.countNonZero(mask_red) > self.GREEN_THRESHOLD_AREA

        # --- Detecção de marcadores verdes e direção ---
        green_direction = None
        green_centroids = []
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_green:
            area = cv2.contourArea(cnt)
            if area > self.GREEN_THRESHOLD_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    gx = int(M["m10"] / M["m00"])
                    gy = int(M["m01"] / M["m00"])
                    green_centroids.append((gx, gy))

        if green_centroids:
            left = any(gx < self.CENTER_X - 50 for gx, _ in green_centroids)
            right = any(gx > self.CENTER_X + 50 for gx, _ in green_centroids)
            if left and right:
                # Dois marcadores verdes, um de cada lado da linha -> retorno de 180°
                green_direction = "uturn"
            elif left:
                green_direction = "left"
            elif right:
                green_direction = "right"

        obstacle = self.detect_obstacle(mask_black)

        # Calcula a curvatura usando regressão polinomial de 2º grau
        curvature = 0
        # Coordenadas Y aproximadas do centro de cada ROI
        roi_y = {"bottom": 230, "middle": 190, "top": 150}
        valid_points = [(roi_y[name], x) for name, x in centroids.items() if x != -1]
        if len(valid_points) >= 3:
            ys, xs = zip(*valid_points)
            coeffs = np.polyfit(ys, xs, 2)
            a, b, _ = coeffs
            y_eval = ys[0]  # avalia a curvatura próximo ao robô
            dxdy = 2 * a * y_eval + b
            ddxdy = 2 * a
            if abs(ddxdy) > 1e-6:
                radius = ((1 + dxdy**2) ** 1.5) / abs(ddxdy)
                curvature = 1 / (radius + 1)  # Normaliza para 0-1
        elif len(valid_points) >= 2:
            # Com poucos pontos, assume-se curva suave
            curvature = 0

        # Contagem de pixels na máscara preta
        pixel_count = cv2.countNonZero(mask_black)

        return (
            cx,
            red_detected,
            obstacle,
            intersection,
            centroids,
            curvature,
            mask_black,
            pixel_count,
            green_direction,
            green_centroids,
        )

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
        elif color_name == 'white':
            self.config['hsv_white']['lower'] = lower_bound
            self.config['hsv_white']['upper'] = upper_bound
            self.log(f"Nova calibração para BRANCO: {lower_bound} a {upper_bound}")

    def detect_obstacle(self, mask_black):
        """ Detecta obstáculos na pista. """
        obstacle_roi = mask_black[self.OBSTACLE_REGION_Y - 10:self.OBSTACLE_REGION_Y + 10, self.CENTER_X - 40:self.CENTER_X + 40]
        contours, _ = cv2.findContours(obstacle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.OBSTACLE_MIN_AREA:
                return True
        return False
