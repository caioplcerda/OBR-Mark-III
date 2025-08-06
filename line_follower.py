import time
from collections import deque
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

        # Histórico para suavização do erro
        self.error_buffer = deque(maxlen=5)
        self.last_error = 0
        self.last_base_speed = self.BASE_SPEED
        self.gap_frames = 0
        self.GAP_HOLD_FRAMES = 5

    def follow_line(self, frame):
        """ Executa a lógica de seguimento de linha para um único frame. """
        # detect_line_features agora retorna 10 valores, incluindo gap e curvatura
        cx, _, _, _, _, gap, centroids, curvature, _, _ = self.vision.detect_line_features(frame)
        status = ""

        # Se a linha foi totalmente perdida e não estamos em modo de pular gap
        if cx == -1 and not (gap or self.gap_frames > 0):
            status = "Linha perdida. Parando."
            self.hardware_control.stop()
            return status, self.path_history

        # Se detectamos um gap, manter direção por alguns frames
        if cx == -1 and (gap or self.gap_frames > 0):
            if gap:
                self.gap_frames = self.GAP_HOLD_FRAMES
            else:
                self.gap_frames -= 1
            self.hardware_control.set_motor_speed(self.last_base_speed, self.last_error)
            status = "Gap detectado. Mantendo direção."
            return status, self.path_history

        # --- Velocidade Adaptativa ---
        max_speed = 80
        min_speed = 40
        # Curvatura normalizada entre 0 e 1 -> reduz proporcionalmente
        speed_reduction = curvature * (max_speed - min_speed)
        base_speed = max_speed - speed_reduction
        base_speed = max(min_speed, min(max_speed, base_speed))

        # Look-ahead adaptativo
        error = self.vision.CENTER_X - cx
        if centroids['middle'] != -1:
            future_error = self.vision.CENTER_X - centroids['middle']
            error = (error + future_error) / 2

        # Suavização do erro com média móvel
        self.error_buffer.append(error)
        filtered_error = sum(self.error_buffer) / len(self.error_buffer)

        self.last_error = filtered_error
        self.last_base_speed = base_speed

        self.hardware_control.set_motor_speed(base_speed, filtered_error)

        # Atualiza o histórico do caminho para visualização
        self.path_history.append((cx, 220))
        if len(self.path_history) > self.PATH_HISTORY_LENGTH:
            self.path_history.pop(0)
        status = f"Error: {int(filtered_error)}"

        return status, self.path_history
