import cv2
import time
import threading
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import numpy as np
from hardware_control import HardwareControl
from vision import Vision
from line_follower import LineFollower
from rescue import Rescue
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
        self.line_follower = LineFollower(self.hardware, self.vision)
        self.rescue = Rescue(self.hardware, self.vision, self.picam2, self.log, self.update_stream_data)

        self.state = "WAITING"

    def update_stream_data(self, frame, mask, path_history, speeds, status_data):
        with SHARED_STATE['stream_lock']:
            s_data = SHARED_STATE['stream_data']
            s_data['last_frame'] = frame.copy()
            s_data['last_mask'] = mask.copy()
            s_data['path_history'] = list(path_history)
            s_data['motor_speeds'] = dict(speeds)
            s_data['status_data'] = dict(status_data)

    def run(self):
        try:
            while True:
                frame_4chan = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_4chan, cv2.COLOR_RGBA2BGR)

                path_history = []
                status_data = {"fsm_state": self.state}
                mask = np.zeros((480, 640), dtype=np.uint8)

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
                    cx, silver, red, obstacle, intersection, centroids, curvature, mask, pixel_count = self.vision.detect_line_features(frame)
                    status, path_history = self.line_follower.follow_line(frame, curvature)
                    if red: self.state = "FINISHING"
                    elif silver: self.state = "RESCUE"
                    elif intersection: self.state = "INTERSECTION"
                    elif obstacle: self.state = "AVOIDING_OBSTACLE"
                    status_data.update({ "pixel_count": pixel_count, "curvature": curvature })

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

                elif self.state == "RESCUE":
                    self.log("Entrando no modo de resgate.")
                    self.rescue.execute_rescue()
                    self.state = "FINISHING"

                elif self.state == "CALIBRATING":
                    self.log("Modo de Calibração Ativo.")
                    self.hardware.stop()

                    # Verifica se há uma requisição de calibração
                    cal_req = SHARED_STATE.get('calibration_request')
                    if cal_req:
                        self.vision.calibrate_by_click(frame, cal_req['x'], cal_req['y'], cal_req['color'])
                        SHARED_STATE['calibration_request'] = None # Limpa a requisição

                    _, _, _, _, _, _, _, mask, pixel_count = self.vision.detect_line_features(frame)
                    status_data.update({"pixel_count": pixel_count})
                    time.sleep(0.1)

                elif self.state == "FINISHING":
                    self.log("Linha de chegada detectada. Finalizando.")
                    self.hardware.stop()
                    time.sleep(5)
                    break

                speeds = {"left": self.hardware.last_left_speed, "right": self.hardware.last_right_speed}
                self.update_stream_data(frame, mask, path_history, speeds, status_data)
                time.sleep(0.05)

        except KeyboardInterrupt:
            self.hardware.cleanup()

if __name__ == '__main__':
    from web_stream import log
    robot = Robot(log_function=log)
    stream_thread = threading.Thread(target=run_stream)
    stream_thread.daemon = True
    stream_thread.start()
    robot.run()
