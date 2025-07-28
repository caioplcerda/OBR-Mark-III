import time
import cv2
from hardware_control import HardwareControl
from vision import Vision

class Rescue:
    """ Classe para a lógica de resgate de bolas. """
    def __init__(self, hardware_control, vision, picam2):
        self.hardware_control = hardware_control
        self.vision = vision
        self.picam2 = picam2

    def open_claw(self):
        """ Abre a garra. """
        self.hardware_control.set_servo_angle(0, 90)

    def close_claw(self):
        """ Fecha a garra. """
        self.hardware_control.set_servo_angle(0, 10)

    def lower_claw(self):
        """ Abaixa a garra. """
        self.hardware_control.set_servo_angle(1, 90)

    def lift_claw(self):
        """ Levanta a garra. """
        self.hardware_control.set_servo_angle(1, 10)

    def release_from_reservoir(self):
        """ Libera as bolas do reservatório. """
        self.hardware_control.set_servo_angle(2, 90)

    def grab_ball(self, pos):
        """ Navega até a bola e a pega. """
        print(f"Navegando para a bola em {pos} e pegando-a.")
        self.lower_claw()
        self.open_claw()
        time.sleep(0.3)
        self.close_claw()
        self.lift_claw()

    def store_in_reservoir(self):
        """ Guarda a bola no reservatório. """
        print("Guardando a bola no reservatório.")
        self.lower_claw()
        self.open_claw()
        time.sleep(0.3)
        self.lift_claw()
        self.close_claw()

    def go_to_area(self, color):
        """ Navega para a área de entrega designada. """
        print(f"Navegando para a área {color}.")
        time.sleep(2)

    def scan_for_panorama(self, duration=6):
        """ Gira o robô para escanear a sala e criar uma imagem panorâmica. """
        print("Iniciando varredura para panorama...")
        self.hardware_control.set_motor_speed(30, -100)  # Gira lentamente

        frames = []
        start_time = time.time()
        while time.time() - start_time < duration:
            frame = self.picam2.capture_array()
            frames.append(frame)
            time.sleep(0.2)

        self.hardware_control.stop()
        print("Varredura concluída. Tentando costurar imagens...")

        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(frames)

        if status == cv2.Stitcher_OK:
            print("Panorama criado com sucesso.")
            return panorama
        else:
            print("Falha ao criar panorama. Usando o último frame como fallback.")
            return frames[-1] if frames else None

    def execute_rescue(self):
        """ Executa a sequência completa de resgate. """
        panorama = self.scan_for_panorama()
        if panorama is None:
            print("Nenhuma imagem para processar no resgate.")
            return

        detected_balls = self.vision.detect_balls(panorama)

        silver_balls = [b for b in detected_balls if b['tipo'] == 'prata']
        black_balls = [b for b in detected_balls if b['tipo'] == 'preta']

        # Pega as bolas prateadas e guarda no reservatório
        for b in silver_balls[:2]:
            self.grab_ball(b['pos'])
            self.store_in_reservoir()

        # Pega a bola preta e a mantém na garra
        if black_balls:
            self.grab_ball(black_balls[0]['pos'])

        # Entrega as bolas
        self.go_to_area("verde")
        self.release_from_reservoir()

        self.go_to_area("vermelha")
        self.open_claw()
        self.lower_claw()
        time.sleep(0.5)
        self.close_claw()
        self.lift_claw()
