# Sistema de Vigilância Distribuído - Guia de Instalação e Execução

## Visão Geral
Sistema distribuído de vigilância com câmara Sony IMX500 no Raspberry Pi que detecta pessoas automaticamente e comunica com PC Windows via Ethernet.

## Arquitetura
- **Raspberry Pi (192.168.2.2)**: Detecção de pessoas com YOLOv8n, captura automática, API Flask
- **PC Windows (192.168.2.1)**: Interface gráfica, recepção de notificações, visualização

## Configuração de Rede

### 1. Configurar IPs Estáticos

#### Raspberry Pi:
```bash
sudo nano /etc/dhcpcd.conf
```
Adicionar:
```
interface eth0
static ip_address=192.168.2.2/24
static routers=192.168.2.1
```

#### Windows PC:
1. Painel de Controle → Rede e Internet → Central de Rede e Partilha
2. Alterar definições do adaptador
3. Ethernet → Propriedades → Protocolo IP Versão 4
4. Configurar:
   - IP: 192.168.2.1
   - Máscara: 255.255.255.0

### 2. Configurar Firewall Windows
```cmd
# Executar como Administrador
netsh advfirewall firewall add rule name="Surveillance_System" dir=in action=allow protocol=TCP localport=5001
```

## Instalação

### Raspberry Pi

1. **Atualizar sistema:**
```bash
sudo apt update && sudo apt upgrade -y
```

2. **Instalar dependências do sistema:**
```bash
sudo apt install python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 -y
```

3. **Criar ambiente virtual:**
```bash
cd /home/pi/DeepVision-Raspberry/Raspberry-Script
python3 -m venv surveillance_env
source surveillance_env/bin/activate
```

4. **Instalar dependências Python:**
```bash
pip install -r requirements_pi.txt
```

5. **Configurar câmara (se usar IMX500):**
```bash
# Ativar câmara
sudo raspi-config
# Interface Options → Camera → Enable

# Para libcamera (IMX500)
sudo apt install libcamera-apps libcamera-dev -y
```

### Windows PC

1. **Instalar Python 3.8+** (se não instalado):
   - Download de python.org
   - Marcar "Add Python to PATH"

2. **Abrir Command Prompt como Administrador:**
```cmd
cd C:\Users\Avarynx\Desktop\DeepVision-Raspberry\Windows-Script
```

3. **Criar ambiente virtual:**
```cmd
python -m venv surveillance_env
surveillance_env\Scripts\activate
```

4. **Instalar dependências:**
```cmd
pip install -r requirements_pc.txt
```

## Execução

### 1. Iniciar Raspberry Pi Agent

```bash
cd /home/pi/DeepVision-Raspberry/Raspberry-Script
source surveillance_env/bin/activate
python pi_agent.py
```

**Saída esperada:**
```
=== Raspberry Pi Agent Starting ===
PI IP: 192.168.2.2:5000
PC IP: 192.168.2.1:5001
Confidence threshold: 0.6
Camera initialized successfully
YOLO model loaded successfully
Starting detection loop...
Starting Flask server...
```

### 2. Iniciar PC Agent

```cmd
cd C:\Users\Avarynx\Desktop\DeepVision-Raspberry\Windows-Script
surveillance_env\Scripts\activate
python pc_agent.py
```

**Interface gráfica irá abrir com:**
- Status do Raspberry Pi
- Botões de controle
- Visualização da última captura
- Log de eventos

## Funcionalidades

### Raspberry Pi Agent (`pi_agent.py`)

**APIs Disponíveis:**
- `GET http://192.168.2.2:5000/last.jpg` - Última imagem capturada
- `POST http://192.168.2.2:5000/capture` - Forçar captura manual
- `GET http://192.168.2.2:5000/health` - Status do sistema
- `GET http://192.168.2.2:5000/stats` - Estatísticas

**Funcionalidades:**
- Detecção automática de pessoas (confiança ≥ 0.6)
- Captura automática com intervalo mínimo de 2 segundos
- Notificação HTTP para PC Windows
- Tolerância a falhas (continua funcionando se PC offline)
- Logs detalhados

### PC Agent (`pc_agent.py`)

**Interface Gráfica:**
- **Status Raspberry Pi**: Online/Offline em tempo real
- **Contador de Capturas**: Número de imagens recebidas
- **Captura Manual**: Botão para forçar captura
- **Atualizar Imagem**: Buscar última imagem do Pi
- **Verificar Status**: Testar conexão com Pi
- **Log de Eventos**: Histórico de atividade
- **Visualização**: Última imagem capturada

**Servidor de Notificações:**
- Recebe POST em `http://192.168.2.1:5001/event`
- Atualiza interface automaticamente
- Salva imagens em `incoming/`

## Configurações Avançadas

### Alterar Sensibilidade de Detecção
No `pi_agent.py`, linha 25:
```python
CONFIDENCE_THRESHOLD = 0.6  # Alterar entre 0.0 e 1.0
```

### Alterar Intervalo de Captura
No `pi_agent.py`, linha 27:
```python
CAPTURE_INTERVAL = 2  # Segundos entre capturas
```

### Alterar Resolução da Câmara
No `pi_agent.py`, linhas 31-32:
```python
CAMERA_WIDTH = 640   # Pixels
CAMERA_HEIGHT = 480  # Pixels
```

## Resolução de Problemas

### Câmara não detectada
```bash
# Verificar dispositivos
ls /dev/video*

# Testar câmara
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERRO')"
```

### Erro de conexão entre dispositivos
```bash
# No Raspberry Pi, testar conectividade
ping 192.168.2.1

# No Windows, testar
ping 192.168.2.2
```

### Firewall bloqueia conexão
```cmd
# Windows - desativar temporariamente
netsh advfirewall set allprofiles state off

# Reativar depois de testar
netsh advfirewall set allprofiles state on
```

### Modelo YOLO não carrega
```bash
# Verificar espaço em disco
df -h

# Download manual do modelo
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Estrutura de Ficheiros Gerada

```
DeepVision-Raspberry/
├── Raspberry-Script/
│   ├── pi_agent.py
│   ├── requirements_pi.txt
│   ├── surveillance_env/
│   ├── captures/           # Imagens capturadas
│   ├── pi_agent.log       # Log do Pi
│   └── yolov8n.pt         # Modelo YOLO (download automático)
└── Windows-Script/
    ├── pc_agent.py
    ├── requirements_pc.txt
    ├── surveillance_env/
    ├── incoming/           # Imagens recebidas
    └── pc_agent.log       # Log do PC
```

## Execução como Serviço (Opcional)

### Raspberry Pi - Systemd Service
```bash
sudo nano /etc/systemd/system/surveillance.service
```

```ini
[Unit]
Description=Surveillance Pi Agent
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/DeepVision-Raspberry/Raspberry-Script
Environment=PATH=/home/pi/DeepVision-Raspberry/Raspberry-Script/surveillance_env/bin
ExecStart=/home/pi/DeepVision-Raspberry/Raspberry-Script/surveillance_env/bin/python pi_agent.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable surveillance.service
sudo systemctl start surveillance.service
```

### Windows - Executar no Arranque
1. Criar ficheiro `start_surveillance.bat`:
```batch
@echo off
cd /d "C:\Users\Avarynx\Desktop\DeepVision-Raspberry\Windows-Script"
call surveillance_env\Scripts\activate
python pc_agent.py
```

2. Adicionar ao arranque:
   - Windows + R → `shell:startup`
   - Copiar `start_surveillance.bat` para esta pasta

## Testes de Funcionamento

1. **Teste de Conectividade:**
   - Verificar se ambos os agentes iniciam sem erro
   - Status do Pi deve aparecer "Online" no PC

2. **Teste de Captura Manual:**
   - Clicar "Captura Manual" na interface
   - Verificar se imagem aparece na interface
   - Confirmar log "Captura manual realizada com sucesso"

3. **Teste de Detecção Automática:**
   - Posicionar pessoa em frente à câmara
   - Aguardar detecção automática
   - Verificar notificação no PC e nova imagem

4. **Teste de Tolerância a Falhas:**
   - Parar PC Agent
   - Verificar se Pi Agent continua funcionando
   - Reiniciar PC Agent - deve reconectar automaticamente

## Contacto e Suporte

Para problemas específicos:
1. Verificar logs em `pi_agent.log` e `pc_agent.log`
2. Testar conectividade de rede
3. Verificar configurações de firewall
4. Confirmar que todas as dependências estão instaladas
