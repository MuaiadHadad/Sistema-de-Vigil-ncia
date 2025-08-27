#!/usr/bin/env python3
"""
PC Agent - Distributed Surveillance System
Windows GUI application that receives notifications from Raspberry Pi
Displays captured images and provides manual capture functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import threading
import time
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image, ImageTk
import io
import logging

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================
PC_IP = "192.168.2.1"
PI_IP = "192.168.2.2"
PC_PORT = 5001
PI_PORT = 5000

# UI Settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
IMAGE_DISPLAY_SIZE = (400, 300)
REFRESH_INTERVAL = 5000  # milliseconds
CONNECTION_TIMEOUT = 5  # seconds

# Local storage
INCOMING_DIR = "Windows-Script/incoming"

# =============================================================================
# FLASK SERVER FOR RECEIVING NOTIFICATIONS
# =============================================================================
flask_app = Flask(__name__)
flask_app.logger.setLevel(logging.ERROR)  # Reduce Flask logging

# Global reference to GUI
gui_instance = None

@flask_app.route('/event', methods=['POST'])
def receive_event():
    """Receive notification from Raspberry Pi"""
    global gui_instance

    try:
        data = request.get_json()
        if data and 'path' in data:
            image_path = data['path']

            # Notify GUI in thread-safe way
            if gui_instance:
                gui_instance.root.after(0, gui_instance.on_new_capture, image_path)

            return jsonify({"status": "received"})
        else:
            return jsonify({"error": "Invalid data"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/health', methods=['GET'])
def pc_health():
    """PC health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gui_active": gui_instance is not None
    })

# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================
class SurveillanceGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sistema de Vigilância - PC Agent")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(True, True)

        # Variables
        self.capture_count = 0
        self.last_image = None
        self.pi_online = False

        # Create incoming directory
        os.makedirs(INCOMING_DIR, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Create GUI elements
        self.create_widgets()

        # Start periodic tasks
        self.start_periodic_updates()

        # Log startup
        self.log_message("=== PC Agent Iniciado ===")
        self.log_message(f"PC IP: {PC_IP}:{PC_PORT}")
        self.log_message(f"PI IP: {PI_IP}:{PI_PORT}")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pc_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Sistema de Vigilância Distribuído",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Left panel - Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Status indicators
        status_frame = ttk.Frame(controls_frame)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(status_frame, text="Status Raspberry Pi:").grid(row=0, column=0, sticky=tk.W)
        self.pi_status_label = ttk.Label(status_frame, text="Verificando...",
                                        foreground="orange")
        self.pi_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(status_frame, text="Capturas Recebidas:").grid(row=1, column=0, sticky=tk.W)
        self.capture_count_label = ttk.Label(status_frame, text="0")
        self.capture_count_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        # Buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.manual_capture_btn = ttk.Button(buttons_frame, text="Captura Manual",
                                           command=self.manual_capture)
        self.manual_capture_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.refresh_btn = ttk.Button(buttons_frame, text="Atualizar Imagem",
                                    command=self.refresh_image)
        self.refresh_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.check_health_btn = ttk.Button(buttons_frame, text="Verificar Status",
                                         command=self.check_pi_health)
        self.check_health_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        buttons_frame.columnconfigure(0, weight=1)

        # Log area
        log_frame = ttk.LabelFrame(controls_frame, text="Log de Eventos", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        controls_frame.rowconfigure(2, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Right panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="Última Captura", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Image display
        self.image_label = ttk.Label(image_frame, text="Nenhuma imagem disponível",
                                   anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # Image info
        self.image_info_label = ttk.Label(image_frame, text="", anchor=tk.CENTER)
        self.image_info_label.grid(row=1, column=0, pady=(10, 0))

    def log_message(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        def update_log():
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)

        # Ensure thread safety
        if threading.current_thread() == threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)

        # Also log to file
        self.logger.info(message)

    def check_pi_health(self):
        """Check Raspberry Pi health status"""
        def check_health():
            try:
                response = requests.get(
                    f"http://{PI_IP}:{PI_PORT}/health",
                    timeout=CONNECTION_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    self.pi_online = True

                    status_text = f"Online - {data.get('captures', 0)} capturas"
                    self.root.after(0, lambda: self.pi_status_label.config(
                        text=status_text, foreground="green"))

                    self.log_message("Status Pi: Online")

                else:
                    self.pi_online = False
                    self.root.after(0, lambda: self.pi_status_label.config(
                        text="Erro de resposta", foreground="red"))

            except requests.exceptions.RequestException:
                self.pi_online = False
                self.root.after(0, lambda: self.pi_status_label.config(
                    text="Offline", foreground="red"))
                self.log_message("Status Pi: Offline")

        threading.Thread(target=check_health, daemon=True).start()

    def manual_capture(self):
        """Request manual capture from Raspberry Pi"""
        def capture():
            try:
                self.log_message("Solicitando captura manual...")

                response = requests.post(
                    f"http://{PI_IP}:{PI_PORT}/capture",
                    timeout=CONNECTION_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        self.log_message("Captura manual realizada com sucesso")
                        # Refresh image after a short delay
                        self.root.after(1000, self.refresh_image)
                    else:
                        self.log_message("Erro na captura manual")
                else:
                    self.log_message(f"Erro no servidor: {response.status_code}")

            except requests.exceptions.RequestException as e:
                self.log_message(f"Erro de conexão: {str(e)}")

        threading.Thread(target=capture, daemon=True).start()

    def refresh_image(self):
        """Refresh the displayed image from Raspberry Pi"""
        def fetch_image():
            try:
                response = requests.get(
                    f"http://{PI_IP}:{PI_PORT}/last.jpg",
                    timeout=CONNECTION_TIMEOUT
                )

                if response.status_code == 200:
                    # Load and resize image
                    image = Image.open(io.BytesIO(response.content))
                    image.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)

                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(image)

                    # Update UI in main thread
                    def update_ui():
                        self.image_label.config(image=photo, text="")
                        self.image_label.image = photo  # Keep reference

                        # Update image info
                        info_text = f"Tamanho: {image.size[0]}x{image.size[1]}\nAtualizado: {datetime.now().strftime('%H:%M:%S')}"
                        self.image_info_label.config(text=info_text)

                        self.log_message("Imagem atualizada")

                    self.root.after(0, update_ui)

                    # Save image locally
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                    filename = f"captura_{timestamp}.jpg"
                    filepath = os.path.join(INCOMING_DIR, filename)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                elif response.status_code == 404:
                    self.log_message("Nenhuma imagem disponível no Pi")
                else:
                    self.log_message(f"Erro ao buscar imagem: {response.status_code}")

            except requests.exceptions.RequestException as e:
                self.log_message(f"Erro de conexão ao buscar imagem: {str(e)}")

        threading.Thread(target=fetch_image, daemon=True).start()

    def on_new_capture(self, image_path):
        """Handle notification of new capture from Pi"""
        self.capture_count += 1
        self.capture_count_label.config(text=str(self.capture_count))

        filename = os.path.basename(image_path)
        self.log_message(f"Nova captura recebida: {filename}")

        # Automatically refresh image
        self.refresh_image()

    def start_periodic_updates(self):
        """Start periodic background tasks"""
        def periodic_health_check():
            self.check_pi_health()
            # Schedule next check
            self.root.after(10000, periodic_health_check)  # Every 10 seconds

        def periodic_image_refresh():
            if self.pi_online:
                self.refresh_image()
            # Schedule next refresh
            self.root.after(REFRESH_INTERVAL, periodic_image_refresh)

        # Start periodic tasks
        self.root.after(1000, periodic_health_check)
        self.root.after(2000, periodic_image_refresh)

    def on_closing(self):
        """Handle application closing"""
        self.log_message("=== PC Agent Encerrando ===")
        self.root.destroy()

    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

# =============================================================================
# FLASK SERVER THREAD
# =============================================================================
def start_flask_server():
    """Start Flask server in background thread"""
    try:
        flask_app.run(
            host='127.0.0.1',
            port=PC_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"Flask server error: {e}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function"""
    global gui_instance

    print("=== PC Agent Starting ===")
    print(f"PC IP: {PC_IP}:{PC_PORT}")
    print(f"PI IP: {PI_IP}:{PI_PORT}")

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()

    # Small delay to let Flask start
    time.sleep(1)

    # Create and run GUI
    try:
        gui_instance = SurveillanceGUI()
        gui_instance.run()
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Erro", f"Erro na aplicação: {e}")
    finally:
        print("=== PC Agent Stopped ===")

if __name__ == "__main__":
    main()
