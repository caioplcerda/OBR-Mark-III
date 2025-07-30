import time
import cv2
from hardware_control import HardwareControl
from vision import Vision

class Rescue:
    """ Classe para a lógica de resgate de bolas. """
    def __init__(self, hardware_control, vision, picam2, log_function, update_frame_function):
        self.hardware = hardware_control
        self.vision = vision
        self.picam2 = picam2
        self.log = log_function
        self.update_live_frame = update_frame_function

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
        """ Gira o robô para escanear a sala, detectando bolas e desenhando no frame. """
        self.log("Iniciando varredura da sala de resgate.")
        self.hardware.set_motor_speed(30, -100)

        all_detected_balls = []
        frames_for_panorama = []
        start_time = time.time()

        while time.time() - start_time < duration:
            frame_4chan = self.picam2.capture_array()
            frame = cv2.cvtColor(frame_4chan, cv2.COLOR_RGBA2BGR)
            frames_for_panorama.append(frame.copy())

            detected_balls = self.vision.detect_balls(frame)
            for ball in detected_balls:
                all_detected_balls.append(ball)
                # Desenha no frame ao vivo para o stream
                pos = ball['pos']
                color = (0, 255, 255) if ball['tipo'] == 'prata' else (128, 128, 128)
                cv2.circle(frame, pos, 15, color, 2)
                cv2.putText(frame, ball['tipo'], (pos[0] + 10, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Atualiza o stream com o frame anotado
            self.update_live_frame(frame)
            time.sleep(0.1)

        self.hardware_control.stop()
        self.log(f"Varredura concluída. Bolas detectadas: {len(all_detected_balls)}")
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
