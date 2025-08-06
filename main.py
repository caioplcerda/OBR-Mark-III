import cv2
import time
import threading
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import numpy as np
from collections import deque
from hardware_control import HardwareControl
from vision import Vision
import advanced_vision as adv_vision
from web_stream import SHARED_STATE, log, run_stream

class Robot:
    def __init__(self, log_function):
        self.log = log_function
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        self.picam2.start()
        time.sleep(1)

        self.hardware = HardwareControl(SHARED_STATE['config'])
        self.vision = Vision(SHARED_STATE['config'], self.log)

        self.state = "WAITING"
        # State for advanced line follower
        self.line_points = deque(maxlen=20)
        self.last_line_angle = -90.0  # Start by looking straight up (-90 deg)

    def update_stream_data(self, frame, mask=None, path_history=None, speeds=None, status_data=None, derivative_scan=None):
        with SHARED_STATE['stream_lock']:
            s_data = SHARED_STATE['stream_data']
            s_data['last_frame'] = frame.copy()
            # Only update other data if it's provided, making the function more robust
            if mask is not None:
                s_data['last_mask'] = mask.copy()
            if path_history is not None:
                s_data['path_history'] = list(path_history)
            if speeds is not None:
                s_data['motor_speeds'] = dict(speeds)
            if status_data is not None:
                s_data['status_data'] = dict(status_data)
            if derivative_scan is not None:
                s_data['derivative_scan'] = derivative_scan

    def run(self):
        try:
            while True:
                try:
                    frame_4chan = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame_4chan, cv2.COLOR_RGBA2BGR)
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    cal_req = SHARED_STATE.get('calibration_request')
                    if cal_req:
                        self.vision.calibrate_by_click(frame, cal_req['x'], cal_req['y'], cal_req['color'])
                        SHARED_STATE['calibration_request'] = None

                    path_history = []
                    status_data = {"fsm_state": self.state}
                    mask = np.zeros((480, 640), dtype=np.uint8)
                    derivative_data = None

                    if self.state == "WAITING":
                        if SHARED_STATE['start_event'].is_set():
                            self.log("Comando da web recebido, iniciando.")
                            SHARED_STATE['start_event'].clear()
                            self.state = "FOLLOWING_LINE"
                        elif not GPIO.input(self.hardware.START_BUTTON):
                            self.log("Botão físico pressionado, iniciando.")
                            time.sleep(0.5)
                            self.state = "FOLLOWING_LINE"

                    elif self.state == "FOLLOWING_LINE":
                        # --- Advanced Line Detection Sequence ---
                        scan_points = []
                        last_scan_center = None

                        # 1. First scan (scanline) to find the line near the robot
                        scan_y = frame.shape[0] - 60
                        scan_center_0 = (frame.shape[1] // 2, scan_y)
                        scan_radius_0 = frame.shape[1] // 2 # Scan the whole width
                        scandata, start_x = adv_vision.scanline(frame_gray, scan_center_0, scan_radius_0)
                        scan_details = {'center_point': scan_center_0, 'radius': scan_radius_0}
                        p0, deriv0 = adv_vision.find_line_from_scan(scandata, start_x, 'line', scan_details)

                        if p0:
                            scan_points.append(p0)
                            last_scan_center = p0
                            derivative_data = deriv0 # Save first derivative for visualization

                            # 2. Subsequent scans (scancircle) to predict the path
                            num_scans = 4
                            scan_radius = 30
                            look_width = 180
                            current_look_angle = self.last_line_angle

                            for i in range(num_scans):
                                if not last_scan_center: break

                                scandata, angles = adv_vision.scancircle(frame_gray, last_scan_center, scan_radius, current_look_angle, look_width)
                                scan_details = {'center_point': last_scan_center, 'radius': scan_radius}
                                p_next, deriv_next = adv_vision.find_line_from_scan(scandata, angles, 'circle', scan_details)

                                if p_next:
                                    current_look_angle = adv_vision.line_angle_from_points(last_scan_center, p_next)
                                    scan_points.append(p_next)
                                    last_scan_center = p_next
                                else:
                                    break

                        # 3. Update state and calculate motor error
                        if len(scan_points) > 1:
                            self.line_points.extendleft(scan_points)
                            self.last_line_angle = adv_vision.line_angle_from_points(scan_points[0], scan_points[-1])

                            # Use a point further down the path for the final error sent to the controller
                            look_ahead_point_index = min(len(scan_points) - 1, 2)
                            error_final = scan_points[look_ahead_point_index][0] - self.vision.CENTER_X

                            base_speed = 50
                            self.hardware.set_motor_speed(base_speed, error_final)

                            path_history = scan_points
                            status_data.update({"error": error_final, "angle": self.last_line_angle})
                        else:
                            self.hardware.stop()
                            status_data.update({"error": "Line Lost"})

                        # Basic detection for intersections, etc. (can be improved)
                        _, _, red, obstacle, intersection, _, _, _, _, _ = self.vision.detect_line_features(frame)
                        if red:
                            self.state = "FINISHING"

                    elif self.state == "INTERSECTION":
                        self.log("Interseção detectada. Parando e seguindo em frente.")
                        self.hardware.stop()
                        time.sleep(1)
                        self.hardware.set_motor_speed(50, 0)
                        self.state = "FOLLOWING_LINE"

                    elif self.state == "AVOIDING_OBSTACLE":
                        self.log("Obstáculo detectado. Desviando.")
                        self.hardware.set_motor_speed(40, 100)
                        time.sleep(0.5)
                        self.hardware.set_motor_speed(50, 0)
                        time.sleep(1)
                        self.state = "FOLLOWING_LINE"

                    elif self.state == "FINISHING":
                        self.log("Linha de chegada detectada. Finalizando.")
                        self.hardware.stop()
                        time.sleep(5)
                        break

                    speeds = {"left": self.hardware.last_left_speed, "right": self.hardware.last_right_speed}
                    self.update_stream_data(frame, mask, path_history, speeds, status_data, derivative_data)
                    time.sleep(0.05)

                except Exception as e:
                    self.log(f"ERRO NO LAÇO PRINCIPAL: {e}")
                    self.log("O robô irá parar por segurança. Reinicie o programa.")
                    self.hardware.stop()
                    # We could break here, but sleeping allows the stream to continue
                    time.sleep(1)


        except KeyboardInterrupt:
            self.hardware.cleanup()

if __name__ == '__main__':
    from web_stream import log
    robot = Robot(log_function=log)
    stream_thread = threading.Thread(target=run_stream)
    stream_thread.daemon = True
    stream_thread.start()
    robot.run()
