import time
from hardware_control import HardwareControl
from vision import Vision

class LineFollower:
    """ Classe para a lógica de seguimento de linha. """
    def __init__(self, hardware_control, vision):
        self.hardware_control = hardware_control
        self.vision = vision
        self.BASE_SPEED = 50
        self.path_history = []
        self.PATH_HISTORY_LENGTH = 20
        self.OBSTACLE_AVOID_SPEED = 40
        self.OBSTACLE_AVOID_TIME = 0.5

    def follow_line(self, frame, curvature):
        """ Executa a lógica de seguimento de linha para um único frame. """
        # A chamada agora desempacota 9 valores, ignorando os que não são usados aqui.
        cx, _, _, _, _, centroids, _, _, _ = self.vision.detect_line_features(frame)
        status = ""

        if cx != -1:
            # --- Velocidade Adaptativa ---
            # Define uma velocidade máxima e mínima
            max_speed = 80
            min_speed = 40

            # Reduz a velocidade com base na curvatura
            # abs(curvature) é usado porque a direção da curva não importa para a velocidade
            speed_reduction = abs(curvature) * 0.5 # Fator de redução (ajustável)
            base_speed = max_speed - speed_reduction
            base_speed = max(min_speed, min(max_speed, base_speed)) # Garante que a velocidade fique nos limites

            # Look-ahead adaptativo
            error = self.vision.CENTER_X - cx
            if centroids['middle'] != -1:
                future_error = self.vision.CENTER_X - centroids['middle']
                error = (error + future_error) / 2

            self.hardware_control.set_motor_speed(base_speed, error)

            # Atualiza o histórico do caminho para visualização
            self.path_history.append((cx, 220))
            if len(self.path_history) > self.PATH_HISTORY_LENGTH:
                self.path_history.pop(0)
            status = f"Error: {int(error)}"
        else:
            status = "Linha perdida. Parando."
            self.hardware_control.stop()

        return status, self.path_history
