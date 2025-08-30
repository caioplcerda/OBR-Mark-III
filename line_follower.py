import cv2
from hardware_control import HardwareControl
from vision import Vision
from advanced_vision import (
    scanline,
    scancircle,
    find_line_from_scan,
    line_angle_from_points,
)

class LineFollower:
    """ Classe para a lógica de seguimento de linha. """
    def __init__(self, hardware_control, vision):
        self.hardware_control = hardware_control
        self.vision = vision
        self.BASE_SPEED = 50
        self.path_history = []
        self.PATH_HISTORY_LENGTH = 20
        # ângulo de busca inicial (olhando para cima)
        self.last_angle = -90

    def follow_line(self, frame):
        """Executa a lógica de seguimento de linha para um único frame."""
        # A câmera está montada de cabeça para baixo; gira 180 graus
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        path_points = []

        # --- Primeira varredura horizontal próxima ao robô ---
        scan_y = gray.shape[0] - 60
        scan_center = (gray.shape[1] // 2, scan_y)
        scan_radius = gray.shape[1] // 2
        scandata, start_x = scanline(gray, scan_center, scan_radius)
        p0, _ = find_line_from_scan(
            scandata,
            start_x,
            "line",
            {"center_point": scan_center, "radius": scan_radius},
            min_line_width=20,
        )

        if p0 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.path_history

        path_points.append(p0)
        last_center = p0
        look_angle = self.last_angle

        # --- Varreduras circulares sucessivas para prever a trajetória ---
        for _ in range(3):
            scandata, angles = scancircle(
                gray,
                last_center,
                30,
                look_angle,
                180,
            )
            p_next, _ = find_line_from_scan(
                scandata,
                angles,
                "circle",
                {"center_point": last_center, "radius": 30},
                min_line_width=20,
            )
            if not p_next:
                break
            path_points.append(p_next)
            look_angle = line_angle_from_points(last_center, p_next)
            last_center = p_next

        # --- Controle dos motores e histórico do caminho ---
        if len(path_points) > 1:
            self.last_angle = line_angle_from_points(path_points[0], path_points[-1])
            look_ahead_index = min(len(path_points) - 1, 2)
            error = path_points[look_ahead_index][0] - self.vision.CENTER_X
            self.hardware_control.set_motor_speed(self.BASE_SPEED, error)
            status = f"Err: {int(error)} Angle: {int(self.last_angle)}"
        else:
            self.hardware_control.stop()
            status = "Linha perdida. Parando."

        self.path_history.extend(path_points)
        if len(self.path_history) > self.PATH_HISTORY_LENGTH:
            self.path_history = self.path_history[-self.PATH_HISTORY_LENGTH:]

        return status, self.path_history
