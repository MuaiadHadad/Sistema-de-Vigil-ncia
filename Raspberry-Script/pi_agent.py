#!/usr/bin/env python3
"""
Raspberry Pi Agent - Distributed Surveillance System
Detects FACES using OpenCV Haar Cascades and Sony IMX500 camera
Communicates with Windows PC via HTTP
"""

import cv2
import os
import time
import json
import requests
import threading
from datetime import datetime
from flask import Flask, send_file, jsonify, request
import numpy as np
import logging

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================
PC_IP = "192.168.2.1"
PI_IP = "192.168.2.2"
PC_PORT = 5001
PI_PORT = 5000

FACE_SCALE_FACTOR = 1.1  # How much the image size is reduced at each scale
FACE_MIN_NEIGHBORS = 5   # How many neighbors each candidate rectangle should have to retain it
FACE_MIN_SIZE = (30, 30) # Minimum possible face size
CAPTURE_INTERVAL = 2     # minimum seconds between captures
CAPTURES_DIR = "captures"

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
app = Flask(__name__)
camera = None
face_cascade = None
last_capture_path = None
last_capture_time = 0
capture_count = 0
use_rpicam = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pi_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CAMERA AND MODEL INITIALIZATION
# =============================================================================
def initialize_camera():
    """Initialize AI camera (Sony IMX500) with optimized Raspberry Pi configuration"""
    global camera, use_rpicam

    try:
        # Release any existing camera
        if camera is not None:
            camera.release()
            time.sleep(1)

        # Reset rpicam flag
        use_rpicam = False

        logger.info("Attempting to initialize Raspberry Pi AI camera (Sony IMX500)...")

        # First approach: Try libcamera with AI camera optimized pipeline
        try:
            logger.info("Attempting libcamera approach optimized for AI camera...")

            # Try multiple GStreamer pipelines optimized for AI cameras
            ai_camera_pipelines = [
                # Pipeline 1: Direct libcamera with format conversion
                "libcamerasrc camera-name=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1,format=YUY2 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1",

                # Pipeline 2: Simple libcamera pipeline
                "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1 max-buffers=1",

                # Pipeline 3: IMX500 specific pipeline
                "libcamerasrc sensor-id=0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1 max-buffers=1",

                # Pipeline 4: Basic format
                "libcamerasrc ! videoconvert ! appsink"
            ]

            for i, pipeline in enumerate(ai_camera_pipelines):
                try:
                    logger.info(f"Testing AI camera pipeline {i + 1}/4...")
                    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

                    if camera.isOpened():
                        # Test frame capture with multiple attempts
                        success_count = 0
                        for attempt in range(3):
                            ret, test_frame = camera.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                success_count += 1
                            time.sleep(0.2)

                        if success_count >= 2:
                            logger.info(f"SUCCESS: AI camera working with pipeline {i + 1}")

                            # Configure for AI camera
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency

                            # Get actual properties
                            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = camera.get(cv2.CAP_PROP_FPS)

                            logger.info(f"AI Camera configured: {width}x{height} @ {fps} FPS")

                            # Extended warm-up for AI camera
                            logger.info("AI camera warm-up...")
                            for j in range(5):
                                camera.read()
                                time.sleep(0.3)

                            return True
                        else:
                            camera.release()
                            logger.warning(f"Pipeline {i + 1} opened but frame capture unreliable")
                    else:
                        if camera:
                            camera.release()

                except Exception as e:
                    logger.debug(f"Pipeline {i + 1} failed: {e}")
                    if camera:
                        camera.release()
                    continue

        except Exception as e:
            logger.warning(f"libcamera AI camera approach failed: {e}")
            if camera:
                camera.release()

        # Second approach: Try rpicam-still for AI camera (often more reliable)
        logger.info("Trying rpicam-still approach for AI camera...")
        try:
            import subprocess

            # Test if rpicam tools work with AI camera
            result = subprocess.run(['rpicam-still', '--list-cameras'],
                                    capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout:
                logger.info(f"AI Camera detected by rpicam: {result.stdout.strip()}")

                # Test AI camera capture with rpicam-still
                def test_ai_camera_capture():
                    try:
                        # AI camera optimized capture settings
                        subprocess.run([
                            'rpicam-still',
                            '-o', '/tmp/ai_test_frame.jpg',
                            '--width', '640',
                            '--height', '480',
                            '--timeout', '2000',
                            '--nopreview',  # No preview for headless operation
                            '--immediate'  # Immediate capture
                        ], capture_output=True, timeout=10)

                        if os.path.exists('/tmp/ai_test_frame.jpg'):
                            frame = cv2.imread('/tmp/ai_test_frame.jpg')
                            os.remove('/tmp/ai_test_frame.jpg')
                            if frame is not None and frame.size > 0:
                                return True, frame
                    except Exception as e:
                        logger.debug(f"AI camera test capture failed: {e}")
                    return False, None

                # Test the AI camera
                ret, frame = test_ai_camera_capture()
                if ret and frame is not None:
                    logger.info("SUCCESS: AI camera working with rpicam-still")
                    use_rpicam = True
                    return True
                else:
                    logger.warning("AI camera detected but capture test failed")

        except Exception as e:
            logger.warning(f"rpicam AI camera approach failed: {e}")

        # Third approach: V4L2 with AI camera specific settings
        logger.info("Trying V4L2 approach for AI camera...")

        # AI cameras often work better with specific video devices
        ai_camera_devices = [0, 1]  # AI cameras typically on video0 or video1

        for device_idx in ai_camera_devices:
            try:
                logger.info(f"Testing AI camera on /dev/video{device_idx}...")

                # Configure device for AI camera
                try:
                    import subprocess
                    subprocess.run([
                        'v4l2-ctl', '--device', f'/dev/video{device_idx}',
                        '--set-fmt-video=width=640,height=480,pixelformat=YUYV'
                    ], capture_output=True, timeout=3)
                except:
                    pass

                camera = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)

                if camera.isOpened():
                    # AI camera specific settings
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))

                    # Test capture stability
                    stable_reads = 0
                    for attempt in range(5):
                        ret, frame = camera.read()
                        if ret and frame is not None and frame.size > 0:
                            stable_reads += 1
                        time.sleep(0.2)

                    if stable_reads >= 3:
                        logger.info(f"SUCCESS: AI camera working on /dev/video{device_idx}")

                        # Final AI camera configuration
                        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = camera.get(cv2.CAP_PROP_FPS)

                        logger.info(f"AI Camera V4L2 configured: {width}x{height} @ {fps} FPS")
                        return True
                    else:
                        camera.release()
                        logger.warning(f"AI camera on /dev/video{device_idx} unstable")
                else:
                    if camera:
                        camera.release()

            except Exception as e:
                logger.debug(f"AI camera V4L2 test on /dev/video{device_idx} failed: {e}")
                if camera:
                    camera.release()

        raise Exception("AI camera initialization failed with all methods")

    except Exception as e:
        logger.error(f"AI camera initialization failed: {e}")
        if camera:
            camera.release()
            camera = None
        return False


def read_camera_frame():
    """Read frame from camera with fallback methods"""
    global camera, use_rpicam

    if use_rpicam:
        # Use rpicam-still for capture
        try:
            import subprocess
            subprocess.run([
                'rpicam-still', '-o', '/tmp/current_frame.jpg',
                '--width', '640', '--height', '480', '--timeout', '1000'
            ], capture_output=True, timeout=5)

            if os.path.exists('/tmp/current_frame.jpg'):
                frame = cv2.imread('/tmp/current_frame.jpg')
                os.remove('/tmp/current_frame.jpg')
                return True, frame
        except:
            pass
        return False, None
    else:
        # Use OpenCV VideoCapture
        if camera and camera.isOpened():
            return camera.read()
        return False, None


# =============================================================================
# DETECTION AND CAPTURE FUNCTIONS
# =============================================================================
def initialize_face_cascade():
    """Initialize Haar Cascade for face detection"""
    global face_cascade

    try:
        logger.info("Loading Haar Cascade for face detection...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            logger.error("Failed to load Haar Cascade XML file")
            return False
        else:
            logger.info("✅ Haar Cascade loaded successfully - Ready for face detection!")
            return True

    except Exception as e:
        logger.error(f"Error loading Haar Cascade: {e}")
        return False


def detect_face(frame):
    """Detect faces in frame using OpenCV Haar Cascades with detailed logging"""
    global face_cascade

    try:
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance the image for better face detection
        gray_frame = cv2.equalizeHist(gray_frame)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            logger.info(f"👤 FACE(S) DETECTED! Found {len(faces)} face(s)")

            # Log details about each detected face
            for i, (x, y, w, h) in enumerate(faces):
                face_size = w * h
                logger.info(f"   Face {i+1}: Position({x},{y}) Size({w}x{h}) Area({face_size}px)")

            # Return the largest face (most prominent)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            logger.info(f"🎯 Best face detected: {w}x{h} pixels - Preparing capture...")

            return faces
        else:
            # Only log every 20th scan to avoid spam
            if hasattr(detect_face, 'scan_count'):
                detect_face.scan_count += 1
            else:
                detect_face.scan_count = 1

            if detect_face.scan_count % 20 == 0:
                logger.debug(f"Scanning for faces... No faces detected (scan #{detect_face.scan_count})")

            return []

    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []


def capture_image(frame, manual=False):
    """Save captured frame and notify PC when face detected"""
    global last_capture_path, last_capture_time, capture_count

    try:
        # Create captures directory if it doesn't exist
        os.makedirs(CAPTURES_DIR, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_detected_{timestamp}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        # Save image
        success = cv2.imwrite(filepath, frame)

        if success:
            last_capture_path = filepath
            last_capture_time = time.time()
            capture_count += 1

            capture_type = "MANUAL" if manual else "AUTOMÁTICA (FACE DETECTADA)"
            logger.info(f"📸 IMAGEM CAPTURADA: {filename} - Tipo: {capture_type}")
            logger.info(f"📊 Total de capturas: {capture_count}")

            # Notify PC (non-blocking)
            threading.Thread(
                target=notify_pc,
                args=(filepath,),
                daemon=True
            ).start()

            return filepath
        else:
            logger.error("Falha ao salvar a imagem")
            return None

    except Exception as e:
        logger.error(f"Erro na captura: {e}")
        return None


def notify_pc(image_path):
    """Send notification to PC about new capture"""
    try:
        url = f"http://{PC_IP}:{PC_PORT}/event"
        payload = {"path": image_path}

        response = requests.post(
            url,
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            logger.info("PC notification sent successfully")
        else:
            logger.warning(f"PC notification failed: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to notify PC: {e}")
    except Exception as e:
        logger.error(f"Notification error: {e}")


# =============================================================================
# MAIN DETECTION LOOP
# =============================================================================
def detection_loop():
    """Main detection loop with rpicam fallback support"""
    global camera, last_capture_time, use_rpicam

    logger.info("Starting detection loop...")

    while True:
        try:
            # Check camera availability differently for rpicam vs OpenCV
            camera_available = False
            if use_rpicam:
                # For rpicam, we don't need a camera object
                camera_available = True
            else:
                # For OpenCV, we need a valid camera object
                camera_available = camera is not None and camera.isOpened()

            if not camera_available:
                logger.error("Camera not available, retrying in 5 seconds...")
                time.sleep(5)
                if initialize_camera():
                    continue
                else:
                    continue

            ret, frame = read_camera_frame()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                time.sleep(1)
                continue

            # Detect faces
            faces = detect_face(frame)

            # Capture if faces detected and enough time has passed
            if len(faces) > 0:
                current_time = time.time()
                if current_time - last_capture_time >= CAPTURE_INTERVAL:
                    capture_image(frame)

            # Small delay to prevent excessive CPU usage
            # Longer delay for rpicam since it's slower
            sleep_time = 1.0 if use_rpicam else 0.1
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Detection loop stopped by user")
            break
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
            time.sleep(1)


# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================
@app.route('/last.jpg', methods=['GET'])
def get_last_image():
    """Return the last captured image"""
    global last_capture_path

    try:
        if last_capture_path and os.path.exists(last_capture_path):
            return send_file(last_capture_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "No image available"}), 404

    except Exception as e:
        logger.error(f"Error serving last image: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/capture', methods=['POST'])
def manual_capture():
    """Force manual capture using appropriate camera method"""
    global camera, use_rpicam

    try:
        ret, frame = read_camera_frame()
        if not ret or frame is None:
            return jsonify({"error": "Failed to capture frame"}), 500

        filepath = capture_image(frame, manual=True)
        if filepath:
            return jsonify({"success": True, "path": filepath})
        else:
            return jsonify({"error": "Capture failed"}), 500

    except Exception as e:
        logger.error(f"Manual capture error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with AI camera support"""
    global camera, face_cascade, capture_count, use_rpicam

    # Check camera status based on mode
    if use_rpicam:
        camera_status = True  # rpicam mode doesn't need camera object
    else:
        camera_status = camera is not None and camera.isOpened()

    # Check face cascade status
    face_cascade_status = face_cascade is not None

    return jsonify({
        "status": "healthy" if camera_status and face_cascade_status else "degraded",
        "camera": "ok" if camera_status else "error",
        "camera_mode": "rpicam" if use_rpicam else "opencv",
        "face_cascade": "ok" if face_cascade_status else "error",
        "captures": capture_count,
        "last_capture": last_capture_path,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        "capture_count": capture_count,
        "last_capture_time": last_capture_time,
        "uptime": time.time() - start_time,
        "captures_dir": CAPTURES_DIR
    })


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function"""
    global start_time
    start_time = time.time()

    logger.info("=== Raspberry Pi Agent Starting ===")
    logger.info(f"PI IP: {PI_IP}:{PI_PORT}")
    logger.info(f"PC IP: {PC_IP}:{PC_PORT}")
    logger.info(f"Face detection scale factor: {FACE_SCALE_FACTOR}")
    logger.info(f"Face detection min neighbors: {FACE_MIN_NEIGHBORS}")
    logger.info(f"Face detection min size: {FACE_MIN_SIZE}")

    # Initialize components
    if not initialize_camera():
        logger.error("Failed to initialize camera. Exiting.")
        return

    # Load Haar Cascade for face detection
    if not initialize_face_cascade():
        logger.error("Failed to initialize face cascade. Exiting.")
        return

    # Start detection loop in separate thread
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    # Start Flask server
    logger.info("Starting Flask server...")
    try:
        app.run(
            host='0.0.0.0',
            port=PI_PORT,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if camera:
            camera.release()
        logger.info("=== Raspberry Pi Agent Stopped ===")


if __name__ == "__main__":
    main()
