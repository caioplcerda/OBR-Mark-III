# Robô Seguidor de Linha OBR 2025

 Este projeto contém o software para um robô autônomo projetado para o desafio da Olimpíada Brasileira de Robótica (OBR) 2025. O robô é capaz de seguir linhas, desviar de obstáculos e superar desafios da arena de forma autônoma.

## Arquitetura do Software

O software é modular e foi desenvolvido em Python, dividido nos seguintes componentes:

 - `main.py`: Orquestrador principal do robô. Contém a máquina de estados que gerencia o comportamento.
- `hardware_control.py`: Camada de abstração para todo o controle de hardware, incluindo motores com encoders, servos e o driver TB6612FNG.
 - `vision.py`: Módulo de visão computacional. Processa as imagens da câmera para detectar a linha e obstáculos.
 - `line_follower.py`: Implementa a lógica de seguimento de linha, utilizando um controle PID e look-ahead adaptativo.
- `web_stream.py`: Um servidor web (Flask) que fornece um stream de vídeo ao vivo e uma interface para calibração de parâmetros.

## Recursos Implementados

### Navegação e Seguimento de Linha
- **Controle PID com Encoders:** Controle preciso de velocidade e distância utilizando um controlador PID que leva em conta a leitura de encoders ópticos em cada roda.
- **Visão Lateral e Look-Ahead:** A câmera é montada de lado (90 graus), permitindo uma visão mais ampla e distante da pista. O software corrige a rotação da imagem e usa uma projeção look-ahead para antecipar curvas.
- **Detecção de Desafios:** Lógica para identificar e transpor desafios como interseções, gaps e obstáculos.
- **Suporte a linhas grossas:** A visão computacional reconhece pistas com até 20 mm de largura usando operações morfológicas e contornos.
- **Marcadores verdes em interseções:** detecção de áreas verdes para orientar o robô sobre qual caminho seguir.


### Interface Web
- **Stream de Vídeo ao Vivo:** Transmite a visão do robô em tempo real para um navegador web.
- **Calibração Remota:** Permite ajustar os principais parâmetros do robô (PID, limites de cor HSV, velocidades) através da interface web, sem a necessidade de alterar o código diretamente.
- **Sinalização de rota:** O stream destaca marcadores verdes detectados e exibe o caminho planejado.

## Configuração e Uso

### 1. Hardware
- **Placa Controladora:** Raspberry Pi 5 (4GB RAM)
- **Câmera:** Picamera3 (Wide)
- **Driver de Motor:** TB6612FNG
- **Motores:** 2x Motores DC com encoders ópticos
- **Servos:** 3x Servos para a garra e o reservatório

### 2. Instalação

Clone o repositório e instale as dependências:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
pip install -r requirements.txt
```

### 3. Execução

Para iniciar o robô, execute o script principal:

```bash
python main.py
```

### 4. Acessando a Interface Web

Com o robô em funcionamento, acesse a interface de controle pelo navegador em qualquer dispositivo na mesma rede Wi-Fi:

`http://<IP_DO_RASPBERRY_PI>:5000`

O IP do Raspberry Pi pode ser encontrado com o comando `hostname -I`.
