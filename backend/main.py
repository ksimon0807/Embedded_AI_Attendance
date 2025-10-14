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

ESP32_CAM_URL = "http://10.130.118.198"


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
            # Extract name and roll from filename (format: name,roll.png)
            name_roll = os.path.splitext(filename)[0]
            if ',' in name_roll:
                name, roll = name_roll.split(',', 1)
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
    """Endpoint to detect if there's a face in the current frame."""
    stream_url = f"{ESP32_CAM_URL}/capture" 
    try:
        print("[DEBUG] Face detection request received")
        print(f"[DEBUG] Camera URL: {stream_url}")
        
        with urllib.request.urlopen(stream_url, timeout=10) as img_resp:
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is None:
                print("[DEBUG] Failed to decode frame in face detection")
                return jsonify({"status": "error", "message": "Could not retrieve frame from camera."})

            print(f"[DEBUG] Face detection frame shape: {frame.shape}")
            
            # Try both HOG and CNN models for better face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # First try with HOG model (faster)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            print(f"[DEBUG] HOG model found {len(boxes)} faces")
            
            # If no faces found with HOG, try CNN model (more accurate but slower)
            if len(boxes) == 0:
                print("[DEBUG] No faces found with HOG, trying CNN model...")
                boxes = face_recognition.face_locations(rgb_frame, model="cnn")
                print(f"[DEBUG] CNN model found {len(boxes)} faces")
            
            if boxes:
                print(f"[DEBUG] Face detection successful - {len(boxes)} face(s) detected")
                return jsonify({
                    "status": "success", 
                    "face_detected": True, 
                    "face_count": len(boxes),
                    "message": f"Found {len(boxes)} face(s) in the frame"
                })
            else:
                print("[DEBUG] No faces detected in the frame")
                return jsonify({
                    "status": "success", 
                    "face_detected": False, 
                    "face_count": 0,
                    "message": "No faces detected in the current frame"
                })

    except urllib.error.URLError as e:
        print(f"[ERROR] Camera connection error during face detection: {e}")
        return jsonify({"status": "error", "message": "Cannot connect to camera. Please check camera connection."})
    except Exception as e:
        print(f"[ERROR] During face detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "An internal server error occurred during face detection."})


# --- Helper Function to log attendance ---
def log_attendance(name,roll):
    try:
        print(f"[INFO] Logging attendance for: {name}")
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        
        csv_file = f"{current_date}.csv"
        
        print(f"[DEBUG] CSV file path: {csv_file}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")

        # Create file with header if not exists
        if not os.path.exists(csv_file):
            print(f"[DEBUG] Creating new CSV file: {csv_file}")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Roll", "Time"])
                print(f"[DEBUG] CSV header written successfully")

        # Read existing names
        existing_names = []
        try:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                existing_names = [row[0] for row in reader if row]  # Filter out empty rows
                print(f"[DEBUG] Existing names in CSV: {existing_names}")
        except Exception as e:
            print(f"[ERROR] Error reading CSV file: {e}")
            return False

        # Append only if not already present
        if name not in existing_names:
            try:
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    current_time = now.strftime("%H:%M:%S")
                    writer.writerow([name,roll, current_time])
                    print(f"[INFO] Successfully logged attendance for {name} at {current_time}")
                return True
            except Exception as e:
                print(f"[ERROR] Error writing to CSV file: {e}")
                return False
        else:
            print(f"[INFO] {name} already has attendance marked for today")
            return False
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in log_attendance: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Frame Capture Generator for Video Feed ---
def gen_frames():
    stream_url = f"{ESP32_CAM_URL}:81/stream"
    print(f"[DEBUG] Video feed URL: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera stream at {stream_url}")
        return

    print("[INFO] Video feed started successfully")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from video stream")
            time.sleep(1)
            continue
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[WARNING] Failed to encode frame as JPEG")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("[INFO] Video feed ended")

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
        
        # Convert to RGB (use full resolution like detect_face function)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Try both HOG and CNN models for better face detection (same as detect_face)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        print(f"[DEBUG] HOG model found {len(face_locations)} faces")
        
        # If no faces found with HOG, try CNN model (more accurate but slower)
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

        recognized_students = []

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
                        recognized_students.append(record)

                        print(f"[INFO] Attendance logged for {name}")
                    else:
                        print(f"[INFO] {name} already marked attendance today")
                else:
                    print(f"[DEBUG] No match found for face {i+1}")
            else:
                print(f"[DEBUG] No matches found for face {i+1}")

        if recognized_students:
            return jsonify({"status": "success", "recognized": recognized_students})
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
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
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


# --- Main ---
if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
