import cv2
import time
import threading
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from hardware_control import HardwareControl
from vision import Vision
from line_follower import LineFollower
from rescue import Rescue
from web_stream import run_stream, update_frame, config, start_command_received

class Robot:
    """ Classe principal que orquestra o robô. """
    def __init__(self):
        # Inicializa a câmera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 240)}))
        self.picam2.start()
        time.sleep(1)

        # Inicializa os módulos de controle
        self.hardware = HardwareControl(config)
        self.vision = Vision(config)
        self.line_follower = LineFollower(self.hardware, self.vision)
        self.rescue = Rescue(self.hardware, self.vision, self.picam2)

        # Define o estado inicial do robô
        self.state = "WAITING"

    def run(self):
        """ Loop principal do robô, gerenciado por uma máquina de estados. """
        try:
            while True:
                frame = self.picam2.capture_array()

                # --- MÁQUINA DE ESTADOS ---

                if self.state == "WAITING":
                    status = "Aguardando início..."
                    path_history = []

                    # Verifica o botão físico OU o comando da web
                    if not GPIO.input(self.hardware.START_BUTTON) or start_command_received:
                        if start_command_received:
                            print("Comando da web recebido, iniciando percurso!")
                            # Reseta a flag
                            start_command_received = False
                        else:
                            print("Botão pressionado, iniciando percurso!")
                            time.sleep(0.5) # Debounce do botão

                        self.state = "FOLLOWING_LINE"

                elif self.state == "FOLLOWING_LINE":
                    cx, silver_detected, red_detected, obstacle, intersection, _, curvature = self.vision.detect_line_features(frame)
                    status, path_history = self.line_follower.follow_line(frame, curvature)

                    # Verifica as condições para mudar de estado
                    if red_detected:
                        self.state = "FINISHING"
                    elif silver_detected:
                        self.state = "RESCUE"
                    elif intersection:
                        self.state = "INTERSECTION"
                    elif obstacle:
                        self.state = "AVOIDING_OBSTACLE"

                    # Atualiza o frame para o stream
                    cv2.putText(frame, f"State: {self.state} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    update_frame(frame, path_history)

                elif self.state == "INTERSECTION":
                    print("Estado: INTERSECTION")
                    self.hardware.stop()
                    # Lógica de decisão de caminho (ex: seguir em frente)
                    time.sleep(1)
                    self.hardware.set_motor_speed(50, 0) # Erro 0 para seguir reto
                    time.sleep(0.5)
                    self.state = "FOLLOWING_LINE"

                elif self.state == "AVOIDING_OBSTACLE":
                    print("Estado: AVOIDING_OBSTACLE")
                    self.hardware.set_motor_speed(40, 100) # Gira para a esquerda
                    time.sleep(0.5)
                    self.hardware.set_motor_speed(50, 0) # Anda para frente
                    time.sleep(1)
                    self.state = "FOLLOWING_LINE"

                elif self.state == "RESCUE":
                    print("Estado: RESCUE")
                    self.rescue.execute_rescue()
                    self.state = "FINISHING"

                elif self.state == "FINISHING":
                    print("Estado: FINISHING. Parando por 5 segundos...")
                    self.hardware.stop()
                    time.sleep(5)
                    print("Percurso finalizado.")
                    break # Encerra o loop principal

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("[EXIT] Encerrando o programa.")
            self.hardware.cleanup()

if __name__ == '__main__':
    robot = Robot()

    # Inicia o stream de vídeo em uma thread separada
    stream_thread = threading.Thread(target=run_stream)
    stream_thread.daemon = True
    stream_thread.start()

    # Inicia o robô
    robot.run()
