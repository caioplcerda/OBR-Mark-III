import cv2
import math
from hardware_control import HardwareControl
from vision import Vision
from advanced_vision import (
    scanline,
    scancircle,
    find_line_from_scan,
)

class LineFollower:
    """ Classe para a lógica de seguimento de linha. """
    def __init__(self, hardware_control, vision):
        self.hardware_control = hardware_control
        self.vision = vision
        self.BASE_SPEED = 50

        # ==== Variáveis portadas do código RCJ ====
        self.scan_height_reg = 220
        self.scan_radius1_reg = 55
        self.scan_radius2_reg = 18
        self.look_angle_reg = 180
        self.look_width_reg = 160

        self.first_angle = 0
        self.first_scanpoint = (160, self.scan_height_reg)
        self.new_scan_radius1 = 140

        # Histórico dos últimos pontos detectados da linha
        self.line_points = [self.first_scanpoint]
        self.scanpoint = self.first_scanpoint

        self.line_history = []
        self.LINE_HISTORY_LENGTH = 20

    def follow_line(self, frame):
        """Executa a lógica de seguimento de linha para um único frame."""
        # A câmera está montada de cabeça para baixo; gira 180 graus
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lista temporária de pontos encontrados nesta iteração
        frame_line_points = []

        # === ZERO SCAN ===
        scandata, start_x = scanline(
            gray,
            (self.first_scanpoint[0], self.scan_height_reg),
            self.new_scan_radius1,
        )
        p0, _ = find_line_from_scan(
            scandata,
            start_x,
            "line",
            {"center_point": (self.first_scanpoint[0], self.scan_height_reg), "radius": self.new_scan_radius1},
            min_line_width=20,
        )

        if p0 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.line_history

        self.first_scanpoint = p0
        self._update_line_points(p0)
        self.new_scan_radius1 = self.scan_radius1_reg

        # === FIRST SCAN ===
        scandata, angles = scancircle(
            gray,
            self.line_points[0],
            self.scan_radius2_reg,
            self.first_angle,
            self.look_width_reg,
        )
        self.scanpoint = self.line_points[0]
        p1, _ = find_line_from_scan(
            scandata,
            angles,
            "circle",
            {"center_point": self.line_points[0], "radius": self.scan_radius2_reg},
            min_line_width=20,
        )
        if p1 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.line_history
        self._update_line_points(p1)
        self.first_angle = self._lineangle()
        self.first_angle = max(-45, min(45, self.first_angle))

        # === SECOND SCAN ===
        scandata, angles = scancircle(
            gray,
            self.line_points[0],
            self.scan_radius2_reg,
            self.first_angle,
            180,
        )
        self.scanpoint = self.line_points[0]
        p2, _ = find_line_from_scan(
            scandata,
            angles,
            "circle",
            {"center_point": self.line_points[0], "radius": self.scan_radius2_reg},
            min_line_width=20,
        )
        if p2 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.line_history
        self._update_line_points(p2)

        # === THIRD SCAN ===
        next_angle = self._lineangle()
        scandata, angles = scancircle(
            gray,
            self.line_points[0],
            self.scan_radius2_reg,
            next_angle,
            180,
        )
        self.scanpoint = self.line_points[0]
        p3, _ = find_line_from_scan(
            scandata,
            angles,
            "circle",
            {"center_point": self.line_points[0], "radius": self.scan_radius2_reg},
            min_line_width=20,
        )
        if p3 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.line_history
        self._update_line_points(p3)

        P_Error = self.first_scanpoint[0] - self.vision.CENTER_X

        # === FOURTH SCAN ===
        next_angle = self._lineangle()
        scandata, angles = scancircle(
            gray,
            self.line_points[0],
            self.scan_radius2_reg,
            next_angle,
            180,
        )
        self.scanpoint = self.line_points[0]
        p4, _ = find_line_from_scan(
            scandata,
            angles,
            "circle",
            {"center_point": self.line_points[0], "radius": self.scan_radius2_reg},
            min_line_width=20,
        )
        if p4 is None:
            self.hardware_control.stop()
            return "Linha perdida. Parando.", self.line_history
        self._update_line_points(p4)
        I_Error = self._lineangle()

        # Controle dos motores (utiliza P_Error)
        self.hardware_control.set_motor_speed(self.BASE_SPEED, P_Error)
        status = f"P:{int(P_Error)} I:{int(I_Error)}"

        # Histórico de pontos para visualização
        frame_line_points.extend(self.line_points[:4])
        self.line_history.extend(frame_line_points)
        if len(self.line_history) > self.LINE_HISTORY_LENGTH:
            self.line_history = self.line_history[-self.LINE_HISTORY_LENGTH:]

        return status, self.line_history

    def _update_line_points(self, point):
        self.line_points.insert(0, point)
        if len(self.line_points) > 7:
            self.line_points = self.line_points[:7]

    def _lineangle(self):
        if not self.line_points or self.scanpoint is None:
            return 0
        lp = self.line_points[0]
        sp = self.scanpoint
        return math.degrees(math.atan2(lp[0] - sp[0], -(lp[1] - sp[1])))