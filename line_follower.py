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

    def follow_line(self, frame):
        """Executa a lógica de seguimento de linha para um único frame."""
        cx, _, _, _, centroids, _, _, _, _ = self.vision.detect_line_features(frame)
        status = ""

        if cx == -1:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.path_history

        error = self.vision.CENTER_X - cx

        derivative = 0
        if centroids['top'] != -1:
            derivative = cx - centroids['top']
        elif centroids['middle'] != -1:
            derivative = cx - centroids['middle']

        control_error = 0.7 * derivative + 0.3 * error
        self.hardware_control.set_motor_speed(self.BASE_SPEED, control_error)

        self.path_history.append((cx, 220))
        if len(self.path_history) > self.PATH_HISTORY_LENGTH:
            self.path_history.pop(0)

        status = f"Err: {int(error)} Der: {int(derivative)}"
        return status, self.path_history
