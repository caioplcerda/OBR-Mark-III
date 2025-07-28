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

    def follow_line(self, frame):
        """ Executa a lógica de seguimento de linha para um único frame. """
        cx, _, _, _, _, _, centroids = self.vision.detect_line_features(frame)
        status = ""

        if cx != -1:
            # Look-ahead adaptativo: usa o centroide do meio para antecipar curvas
            error = self.vision.CENTER_X - cx
            if centroids['middle'] != -1:
                future_error = self.vision.CENTER_X - centroids['middle']
                error = (error + future_error) / 2  # Média simples para suavizar a resposta

            self.hardware_control.set_motor_speed(self.BASE_SPEED, error)

            # Atualiza o histórico do caminho para visualização
            self.path_history.append((cx, 220))
            if len(self.path_history) > self.PATH_HISTORY_LENGTH:
                self.path_history.pop(0)
            status = f"Error: {int(error)}"
        else:
            status = "Linha perdida. Parando."
            self.hardware_control.stop()

        return status, self.path_history
