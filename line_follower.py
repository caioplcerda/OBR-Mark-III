# line_follower.py (UNIFICADO)
import cv2
import math
from collections import deque

from hardware_control import HardwareControl
from vision import Vision
import rcj2014_port as rcj

class LineFollower:
    """
    Seguidor de linha unificado:
      - RCJ scans (scanline + 3x scancircle) com derivada -> ponto/ângulo
      - Planejamento curto com look-ahead
      - PID em cima de P (offset) e "I" ~ ângulo (feed-forward)
      - Marcadores verdes (left/right/uturn) e linha de chegada (vermelho)
      - Publica tudo para a interface web (derivada, máscara, histórico)
    """
    def __init__(self, hardware: HardwareControl, vision: Vision, shared_state, logger):
        self.hw = hardware
        self.vision = vision
        self.SS = shared_state
        self.log = logger

        self.base_speed = 50
        self.scan_h = 420          # y para o scanline (imagem 480p após giro de 180°)
        self.radius_line = 320     # varrer largura inteira
        self.radius_circle = 22
        self.look_width_deg = 180

        self.first_angle_deg = 0
        self.first_scanpoint = (self.vision.CENTER_X, self.scan_h)
        self.line_points = deque(maxlen=7)
        self.scanpoint = None

    # ------------ loop principal por frame ------------
    def step(self, frame_bgr):
        # A tua câmera está invertida fisicamente: manter giros consistentes
        frame = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # (A) “ZERO SCAN” — linha perto do robô
        scan0, x0 = rcj.scanline(gray, (self.first_scanpoint[0], self.scan_h), self.radius_line)
        p0, deriv0 = rcj.find_line_from_scan(
            scan0, x0, "line",
            {"center_point": (self.first_scanpoint[0], self.scan_h), "radius": self.radius_line},
            min_line_width=12,
        )
        if not p0:
            self.hw.stop()
            return "Linha perdida (zero scan).", {}

        self.first_scanpoint = p0
        self._push_point(p0)

        # (B) “FIRST SCAN” — círculo olhando com ângulo estimado
        scan1, angs1 = rcj.scancircle(gray, self.line_points[0], self.radius_circle, self.first_angle_deg, self.look_width_deg)
        p1, deriv1 = rcj.find_line_from_scan(
            scan1, angs1, "circle",
            {"center_point": self.line_points[0], "radius": self.radius_circle},
            min_line_width=6,
        )
        if not p1:
            self.hw.stop()
            return "Linha perdida (first scan).", {}
        self._push_point(p1)

        self.first_angle_deg = max(-45, min(45, rcj.line_angle_from_points(self.line_points[1], self.line_points[0])))

        # (C) “SECOND/THIRD SCAN” — refina com novo look-ahead
        scan2, angs2 = rcj.scancircle(gray, self.line_points[0], self.radius_circle, self.first_angle_deg, 180)
        p2, deriv2 = rcj.find_line_from_scan(scan2, angs2, "circle",
                                             {"center_point": self.line_points[0], "radius": self.radius_circle}, 6)
        if not p2:
            self.hw.stop()
            return "Linha perdida (second scan).", {}
        self._push_point(p2)

        scan3, angs3 = rcj.scancircle(gray, self.line_points[0], self.radius_circle,
                                      rcj.line_angle_from_points(self.line_points[1], self.line_points[0]), 180)
        p3, deriv3 = rcj.find_line_from_scan(scan3, angs3, "circle",
                                             {"center_point": self.line_points[0], "radius": self.radius_circle}, 6)
        if not p3:
            self.hw.stop()
            return "Linha perdida (third scan).", {}
        self._push_point(p3)

        # (D) Erros de curso (P e “I” ~ ângulo)
        P_err = self.first_scanpoint[0] - self.vision.CENTER_X
        I_err = rcj.line_angle_from_points(self.line_points[1], self.line_points[0]) if len(self.line_points) > 1 else 0.0

        # (E) Ajuste de motores (PID interno do HardwareControl já usa o erro composto)
        err_composto = P_err + 0.6 * I_err     # mistura leve de ângulo (feed-forward)
        self.hw.set_motor_speed(self.base_speed, err_composto)

        # (F) Eventos de pista (verde/vermelho) + máscara para UI
        green_centroids, green_dir, mask_green = rcj.track_green_centroids(
            frame,
            {"lower": self.vision.LOWER_GREEN, "upper": self.vision.UPPER_GREEN},
            area_min=self.vision.GREEN_THRESHOLD_AREA
        )
        red_detected = self._detect_red(frame)

        # (G) Decisões simples: retorno/curvas por marcador
        if green_dir == "uturn":
            self.log("Marcadores verdes em ambos os lados: retorno 180°.")
            self.hw.set_motor_speed(0, 120)
            return "UTurn executado.", self._pack_status(P_err, I_err, mask_green, deriv0)

        # (H) Atualiza UI
        status = self._pack_status(P_err, I_err, mask_green, deriv0,
                                   extras={"green": green_dir, "greens": green_centroids, "red": red_detected})

        if red_detected:
            self.hw.stop()
            return "Chegada detectada.", status

        return "OK", status

    # ------------ helpers ------------
    def _push_point(self, p):
        self.line_points.appendleft(p)
        self.scanpoint = p

    def _detect_red(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.vision.LOWER_RED1, self.vision.UPPER_RED1)
        m2 = cv2.inRange(hsv, self.vision.LOWER_RED2, self.vision.UPPER_RED2)
        mask_red = cv2.add(m1, m2)
        return cv2.countNonZero(mask_red) > self.vision.GREEN_THRESHOLD_AREA

    def _pack_status(self, P_err, I_err, mask, deriv_scan, extras=None):
        # empacota para web_stream.SHARED_STATE
        extras = extras or {}
        return {
            "P_err": float(P_err),
            "I_err": float(I_err),
            "derivative_scan": deriv_scan.tolist() if deriv_scan is not None else None,
            **extras
        }

