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
import web_stream

class Robot:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        self.picam2.start()
        time.sleep(1)

        self.hardware = HardwareControl(web_stream.config)
        self.vision = Vision(web_stream.config)
        self.line_follower = LineFollower(self.hardware, self.vision)
        self.rescue = Rescue(self.hardware, self.vision, self.picam2)

        self.state = "WAITING"

    def run(self):
        try:
            while True:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                path_history = []
                status_data = {"fsm_state": self.state}
                mask = np.zeros((480, 640), dtype=np.uint8)

                if self.state == "WAITING":
                    if not GPIO.input(self.hardware.START_BUTTON) or web_stream.start_event.is_set():
                        if web_stream.start_event.is_set():
                            web_stream.log("Comando da web recebido, iniciando.")
                            web_stream.start_event.clear()
                        else:
                            web_stream.log("Botão físico pressionado, iniciando.")
                            time.sleep(0.5)
                        self.state = "FOLLOWING_LINE"

                elif self.state == "FOLLOWING_LINE":
                    cx, silver, red, obstacle, intersection, centroids, curvature, mask, pixel_count = self.vision.detect_line_features(frame)
                    status, path_history = self.line_follower.follow_line(frame, curvature)

                    if red: self.state = "FINISHING"
                    elif silver: self.state = "RESCUE"
                    elif intersection: self.state = "INTERSECTION"
                    elif obstacle: self.state = "AVOIDING_OBSTACLE"

                    status_data.update({
                        "line_follower_status": status, "silver_detected": silver, "red_detected": red,
                        "obstacle_detected": obstacle, "intersection_detected": intersection, "curvature": curvature,
                        "pixel_count": pixel_count
                    })

                elif self.state == "INTERSECTION":
                    web_stream.log("Interseção detectada. Parando e seguindo em frente.")
                    self.hardware.stop()
                    time.sleep(1)
                    self.hardware.set_motor_speed(50, 0)
                    self.state = "FOLLOWING_LINE"

                elif self.state == "AVOIDING_OBSTACLE":
                    web_stream.log("Obstáculo detectado. Desviando.")
                    self.hardware.set_motor_speed(40, 100)
                    time.sleep(0.5)
                    self.hardware.set_motor_speed(50, 0)
                    time.sleep(1)
                    self.state = "FOLLOWING_LINE"

                elif self.state == "RESCUE":
                    web_stream.log("Entrando no modo de resgate.")
                    self.rescue.execute_rescue()
                    self.state = "FINISHING"

                elif self.state == "FINISHING":
                    web_stream.log("Linha de chegada detectada. Finalizando.")
                    self.hardware.stop()
                    time.sleep(5)
                    break

                speeds = {"left": self.hardware.last_left_speed, "right": self.hardware.last_right_speed}
                web_stream.update_stream_data(frame, mask, path_history, speeds, status_data)
                time.sleep(0.05)

        except KeyboardInterrupt:
            self.hardware.cleanup()

if __name__ == '__main__':
    robot = Robot()
    stream_thread = threading.Thread(target=web_stream.run_stream)
    stream_thread.daemon = True
    stream_thread.start()
    robot.run()
