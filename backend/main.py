import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import urllib.request
import os
from flask import Flask, Response, jsonify
from flask_cors import CORS
import time

from cryptography.fernet import Fernet
# --- Encryption Setup ---
with open("secret.key", "rb") as key_file:
    encryption_key = key_file.read()
fernet = Fernet(encryption_key)


# --- ESP32-CAM URL ---
ESP32_CAM_URL = "http://172.20.10.14"
import cv2
import threading
import time

class CameraStream:
    def __init__(self, url):
        self.url = url
        self.stream_url = f"{url}:81/stream"
        self.capture_url = f"{url}/capture"
        self.cap = None
        self.frame = None
        self.detected_frame = None
        self.running = True
        self.use_stream = True
        self.last_capture_time = 0
        self.capture_interval = 0.1  # Capture every 100ms as fallback

        # Try to open stream first
        try:
            self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                print(f"[WARNING] Stream not available at {self.stream_url}, using capture endpoint")
                self.use_stream = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
        except Exception as e:
            print(f"[WARNING] Failed to open stream: {e}, using capture endpoint")
            self.use_stream = False
            if self.cap:
                self.cap.release()
                self.cap = None

        if self.use_stream:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        else:
            # Use capture endpoint in a thread
            self.thread = threading.Thread(target=self.update_capture, daemon=True)
            self.thread.start()

        # Start detection in another thread
        self.det_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.det_thread.start()

    def update(self):
        """Update frames from stream"""
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.frame = frame
                    else:
                        time.sleep(0.05)
                else:
                    # Stream closed, switch to capture
                    print("[WARNING] Stream closed, switching to capture endpoint")
                    self.use_stream = False
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    # Start capture thread in a new thread to avoid nesting
                    capture_thread = threading.Thread(target=self.update_capture, daemon=True)
                    capture_thread.start()
                    break
            except Exception as e:
                print(f"[ERROR] Error reading stream: {e}")
                # If stream fails repeatedly, switch to capture
                self.use_stream = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                capture_thread = threading.Thread(target=self.update_capture, daemon=True)
                capture_thread.start()
                break

    def update_capture(self):
        """Update frames from capture endpoint"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_capture_time >= self.capture_interval:
                    with urllib.request.urlopen(self.capture_url, timeout=5) as resp:
                        img_np = np.array(bytearray(resp.read()), dtype=np.uint8)
                        frame = cv2.imdecode(img_np, -1)
                        if frame is not None:
                            self.frame = frame
                        self.last_capture_time = current_time
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"[ERROR] Error capturing frame: {e}")
                time.sleep(0.1)
        
    def run_detection(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue

            try:
                frame_copy = self.frame.copy()
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                self.detected_frame = frame_copy
            except Exception as e:
                print(f"[ERROR] Error in detection: {e}")
            time.sleep(0.02)  # small pause to avoid CPU overload

    def get_frame(self):
        # Return detected frame if available, else raw frame
        return self.detected_frame if self.detected_frame is not None else self.frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# Global camera stream instance
camera_stream = None
camera_lock = threading.Lock()

def get_camera_stream():
    global camera_stream
    with camera_lock:
        if camera_stream is None:
            camera_stream = CameraStream(ESP32_CAM_URL)
            print("[INFO] Camera and detection threads started")
        return camera_stream

def gen_frames():
    cam = get_camera_stream()
    frame_timeout = 0
    max_timeout = 50  # Max 5 seconds without frames
    
    while True:
        try:
            frame = cam.get_frame()
            if frame is None:
                frame_timeout += 1
                if frame_timeout > max_timeout:
                    # Try to recreate camera stream
                    print("[WARNING] No frames received, recreating camera stream")
                    with camera_lock:
                        if camera_stream:
                            camera_stream.stop()
                        camera_stream = None
                    cam = get_camera_stream()
                    frame_timeout = 0
                time.sleep(0.1)
                continue

            frame_timeout = 0  # Reset timeout on successful frame
            
            # Resize frame for efficient streaming
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                print("[ERROR] Failed to encode frame")
                time.sleep(0.1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"[ERROR] Error in gen_frames: {e}")
            time.sleep(0.1)

# Helper to get latest frame quickly from the running camera stream
def get_latest_frame(timeout_seconds: float = 2.0):
    cam = get_camera_stream()
    start = time.time()
    while time.time() - start < timeout_seconds:
        frame = cam.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


faces_dir = "faces"

known_face_encodings = []
known_face_names = []
known_face_rolls = []

for filename in os.listdir(faces_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            # Extract name and roll from filename
            name_roll = os.path.splitext(filename)[0]
            if '_' in name_roll:
                name, roll = name_roll.split('_', 1)
                name = name.strip()
                roll = roll.strip()
            else:
                # fallback if no comma found
                name = name_roll
                roll = "Unknown"

            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_face_rolls.append(roll)


# --- Flask Initialization ---
app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/detect_face', methods=['GET'])
def detect_face():
    """Detect faces using the latest in-memory frame for low latency."""
    try:
        frame = get_latest_frame(timeout_seconds=2.0)
        if frame is None:
            return jsonify({
                "status": "success",
                "face_detected": False,
                "face_count": 0,
                "message": "No frame available yet"
            })

        # Downscale for faster CPU inference if large
        h, w = frame.shape[:2]
        target_w = 640
        if w > target_w:
            scale = target_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame, model="hog")

        return jsonify({
            "status": "success",
            "face_detected": len(boxes) > 0,
            "face_count": len(boxes),
            "message": ("Found face(s)" if len(boxes) > 0 else "No faces detected")
        })
    except Exception as e:
        print(f"[ERROR] During face detection: {e}")
        return jsonify({"status": "error", "message": "Face detection failed"})


# --- Helper Function to log attendance ---
def log_attendance(name, roll):
    try:
        print(f"[INFO] Logging attendance for: {name}")
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        csv_file = f"{current_date}.csv"

        print(f"[DEBUG] CSV file path: {csv_file}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")

        # Read existing decrypted entries
        existing_names = []
        if os.path.exists(csv_file):
            with open(csv_file, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"gAAAAA"):
                        try:
                            decrypted = fernet.decrypt(line).decode()
                            existing_names.append(decrypted.split(",")[0])
                        except Exception:
                            continue

        print(f"[DEBUG] Existing names in CSV: {existing_names}")

        # If already present, skip
        if name in existing_names:
            print(f"[INFO] {name} already has attendance marked for today")
            return False

        # Otherwise, append new encrypted entry
        current_time = now.strftime("%H:%M:%S")
        new_line = f"{name},{roll},{current_time}".encode()
        encrypted_line = fernet.encrypt(new_line)

        with open(csv_file, "ab") as f:
            f.write(encrypted_line + b"\n")

        print(f"[INFO] Successfully logged encrypted attendance for {name}")
        return True

    except Exception as e:
        print(f"[ERROR] Unexpected error in log_attendance: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Frame Capture Generator for Video Feed ---

# --- Routes ---

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mark_attendance',methods=['GET', 'POST'])
def mark_attendance():
    stream_url = f"{ESP32_CAM_URL}/capture"
    try:
        print("[DEBUG] Mark attendance request received")
        
        # Capture frame from ESP32
        with urllib.request.urlopen(stream_url, timeout=10) as resp:
            img_np = np.array(bytearray(resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is None:
                print("[DEBUG] Failed to decode frame in mark attendance")
                return jsonify({"status": "error", "message": "Failed to capture frame from camera"})
        
        print(f"[DEBUG] Frame captured successfully, shape: {frame.shape}")
        
        # Convert to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        print(f"[DEBUG] HOG model found {len(face_locations)} faces")
        
        if len(face_locations) == 0:
            print("[DEBUG] No faces found with HOG, trying CNN model...")
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            print(f"[DEBUG] CNN model found {len(face_locations)} faces")

        if len(face_locations) == 0:
            print("[DEBUG] No faces detected in the frame")
            return jsonify({"status": "error", "message": "No face detected in the frame"})
        
        # Get face encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"[DEBUG] Found {len(face_locations)} faces in frame, {len(face_encodings)} encodings generated")
        
        if len(known_face_encodings) == 0:
            return jsonify({"status": "error", "message": "No known faces configured on server"})

        recognized_people = []

        for i, face_encoding in enumerate(face_encodings):
            print(f"[DEBUG] Processing face {i+1}")
            
            # Use a more lenient tolerance for face matching
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            print(f"[DEBUG] Face distances: {face_distances}")
            print(f"[DEBUG] Matches: {matches}")
            
            if True in matches:  # Check if any match is found
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:  # Double check the best match
                    name = known_face_names[best_match_index]
                    print(f"[INFO] Recognized: {name} with distance: {face_distances[best_match_index]}")
                    
                    # Log attendance
                    roll = known_face_rolls[best_match_index]
                    record = log_attendance(name, roll)
                    if record:
                        recognized_people.append({"name": name, "roll": roll, "marked": True})
                        print(f"[INFO] Attendance logged for {name}")
                    else:
                        recognized_people.append({"name": name, "roll": roll, "marked": False})
                        print(f"[INFO] {name} already marked attendance today")
                else:
                    print(f"[DEBUG] No match found for face {i+1}")
            else:
                print(f"[DEBUG] No matches found for face {i+1}")

        if recognized_people:
            any_marked = any(p.get("marked") for p in recognized_people)
            if any_marked:
                names_marked = ", ".join([p["name"] for p in recognized_people if p.get("marked")])
                return jsonify({
                    "status": "success",
                    "recognized": recognized_people,
                    "message": f"Attendance marked for: {names_marked}"
                })
            else:
                names_already = ", ".join([p["name"] for p in recognized_people])
                return jsonify({
                    "status": "success",
                    "recognized": recognized_people,
                    "message": f"Already marked today: {names_already}"
                })
        else:
            return jsonify({"status": "error", "message": "Face detected but person not recognized"})

    except urllib.error.URLError as e:
        print(f"[ERROR] Camera connection error: {e}")
        return jsonify({"status": "error", "message": "Cannot connect to camera. Please check camera connection."})
    except Exception as e:
        print(f"[ERROR] Unexpected error in mark_attendance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"})


@app.route('/get_attendance')
def get_attendance():
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_file = f"{current_date}.csv"

    if not os.path.exists(csv_file):
        return jsonify([])

    rows = []
    with open(csv_file, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if not line.startswith(b"gAAAAA"):
                    continue 
                decrypted = fernet.decrypt(line).decode()
                name, roll, time = decrypted.split(",")
                rows.append({"Name": name, "Roll": roll, "Time": time})
            except Exception as e:
                print(f"[ERROR] Failed to decrypt a line: {e}")
                continue

    return jsonify(rows)


@app.route('/test_camera')
def test_camera():
    """Test endpoint to check camera connectivity and basic functionality."""
    try:
        print("[DEBUG] Camera test request received")
        
        # Test camera capture
        stream_url = f"{ESP32_CAM_URL}/capture"
        print(f"[DEBUG] Testing camera at: {stream_url}")
        
        with urllib.request.urlopen(stream_url, timeout=10) as resp:
            img_np = np.array(bytearray(resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            
            if frame is None:
                return jsonify({
                    "status": "error", 
                    "message": "Camera is accessible but frame decoding failed"
                })
            
            # Test face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            
            return jsonify({
                "status": "success",
                "message": "Camera test successful",
                "frame_shape": frame.shape,
                "faces_detected": len(boxes),
                "camera_url": stream_url
            })
            
    except urllib.error.URLError as e:
        return jsonify({
            "status": "error",
            "message": f"Cannot connect to camera: {str(e)}",
            "camera_url": stream_url
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Camera test failed: {str(e)}"
        })
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flask HTTPS Active</title>
        <style>
            body {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                text-align: center;
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 0.5em;
                text-shadow: 1px 1px 5px rgba(0,0,0,0.4);
            }
            p {
                font-size: 1.2em;
                color: #e0e0e0;
                max-width: 600px;
                line-height: 1.5em;
            }
            .status {
                margin-top: 1.5em;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.9; }
                100% { transform: scale(1); opacity: 1; }
            }
            footer {
                position: absolute;
                bottom: 20px;
                font-size: 0.9em;
                color: #cccccc;
            }
        </style>
    </head>
    <body>
        <h1> Secure Flask Server Running</h1>
        <p>
            Your Flask backend is successfully running with <strong>HTTPS encryption</strong> enabled.<br/>
            All data between the client and server is securely transmitted.
        </p>
        <div class="status">HTTPS Port 5000 Active </div>
        <footer>© 2025 Secure Flask | TLS/SSL Enabled</footer>
    </body>
    </html>
    """



if __name__ == '__main__':
    print("[INFO] Starting Flask server securely with HTTPS...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        ssl_context=('cert.pem', 'key.pem'),
        threaded=True
    )

