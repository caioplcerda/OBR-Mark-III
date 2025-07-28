# OBR 2025 - Robô Seguidor de Linha com Sala de Resgate

Este repositório contém dois módulos principais desenvolvidos para a Olimpíada Brasileira de Robótica 2025:

- `robotfollow.py`: controle avançado de seguidor de linha com câmera lateral e visão computacional (OpenCV).
- `resgate_bolas.py`: lógica completa para sala de resgate com varredura em 360°, detecção de bolas e manipulação com servos.

## Recursos

### `robotfollow.py`
- Controle PID com anti-wind-up
- Projeção look-ahead adaptativa (linha e verde)
- Curvatura adaptativa para curvas suaves
- Detecção de obstáculos físicos reais (como garrafas e caixas)
- Identificação e interpretação correta de cruzamentos
- Stream ao vivo com trajetória e decisão sobre caminho

### `resgate_bolas.py`
- Varredura estilo radar (panorâmica) da sala de resgate
- Detecção de bolas prateadas e preta usando HSV
- Manipulação com garra e reservatório usando 3 servos
- Entrega das bolas nas regiões corretas
- Comandos modulares para integração com seu sistema principal

## Pré-requisitos

- Raspberry Pi 5 (recomendado) com PiCamera 3 Wide
- OpenCV, NumPy, Flask, RPi.GPIO, picamera2 instalados
- Fonte de alimentação confiável (mínimo 3A)
- pigpio (opcional, para controle de encoders com precisão)

## Executando

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/obr2025-robot.git
cd obr2025-robot
