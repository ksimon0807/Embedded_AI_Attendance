import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import urllib.request
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import os
import time

# --- Configuration ---
# IMPORTANT: Update this URL to the IP address shown in your Arduino IDE Serial Monitor
ESP32_CAM_URL = "http://192.168.165.198" 

ENCODINGS_FILE = "encodings.pickle"
STUDENT_DATA_CSV = "student_data.csv"
ATTENDANCE_LOG_CSV = "attendance_log.csv"

# --- Initialization ---
app = Flask(__name__)
CORS(app) 

print("[INFO] Loading face encodings...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"[ERROR] Encodings file not found at '{ENCODINGS_FILE}'. Please run encode_faces.py first.")
    exit()

try:
    student_df = pd.read_csv(STUDENT_DATA_CSV)
except FileNotFoundError:
    print(f"[ERROR] Student data CSV not found at '{STUDENT_DATA_CSV}'. Please create it.")
    exit()

# Initialize attendance log with header if it doesn't exist
if not os.path.exists(ATTENDANCE_LOG_CSV):
    pd.DataFrame(columns=["Name", "Roll", "Timestamp"]).to_csv(ATTENDANCE_LOG_CSV, index=False)


print("[INFO] Face encodings and student data loaded successfully.")

# --- Helper Functions ---

def log_attendance(name):
    """Logs the attendance of a recognized student and checks for duplicates."""
    try:
        print(f"[DEBUG] Attempting to log attendance for: {name}")
        
        # Check if attendance log file exists and has data
        if os.path.exists(ATTENDANCE_LOG_CSV):
            try:
                attendance_log = pd.read_csv(ATTENDANCE_LOG_CSV)
                print(f"[DEBUG] Attendance log loaded with {len(attendance_log)} rows")
                
                # Check if file has data (more than just headers)
                if len(attendance_log) > 0:
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    print(f"[DEBUG] Checking for duplicates on: {today_str}")
                    
                    # Filter entries for today
                    todays_entries = attendance_log[attendance_log['Timestamp'].str.startswith(today_str)]
                    print(f"[DEBUG] Found {len(todays_entries)} entries for today")
                    
                    if name in todays_entries['Name'].values:
                        print(f"[INFO] {name} has already been marked present today.")
                        return {"status": "already_marked", "message": f"{name} has already been marked present today."}
                else:
                    print(f"[DEBUG] Attendance log is empty, proceeding with new entry")
            except Exception as e:
                print(f"[DEBUG] Error reading attendance log, will create new one: {e}")
        else:
            print(f"[DEBUG] Attendance log file doesn't exist, will create new one")

        # Get student info from student data
        print(f"[DEBUG] Looking up student info for: {name}")
        student_info = student_df[student_df['Name'] == name].iloc[0]
        roll = student_info['Roll']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[DEBUG] Student info - Roll: {roll}")

        # Create new attendance entry
        new_entry = pd.DataFrame([{"Name": name, "Roll": roll, "Timestamp": timestamp}])
        print(f"[DEBUG] New entry created: {new_entry.to_dict('records')}")
        
        # Append to attendance log
        new_entry.to_csv(ATTENDANCE_LOG_CSV, mode='a', header=False, index=False)
        print(f"[SUCCESS] Attendance logged for {name} in {ATTENDANCE_LOG_CSV}")
        
        return {
            "status": "success",
            "data": {"name": name, "roll": str(roll)}
        }

    except IndexError:
        print(f"[WARNING] Recognized name '{name}' not found in {STUDENT_DATA_CSV}")
        return {"status": "error", "message": f"Recognized person '{name}' not found in student records."}
    except Exception as e:
        print(f"[ERROR] Failed to log attendance: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"An internal server error occurred while logging: {str(e)}"}

def gen_frames():
    """
    [NEW & IMPROVED] Generator function using OpenCV to robustly capture frames 
    from ESP32-CAM and yield them for streaming.
    """
    stream_url = f"{ESP32_CAM_URL}:81/stream"
    print(f"[INFO] Connecting to camera stream at {stream_url} using OpenCV...")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("[ERROR] Could not open video stream. Please check the URL and camera availability.")
        return

    print("[INFO] Camera stream connected successfully.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from stream. Reconnecting...")
            time.sleep(2) # Wait a moment before retrying
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("[ERROR] Failed to reconnect to the stream. Ending feed.")
                break
            continue

        (flag, encodedImage) = cv2.imencode(".png", frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    
    print("[INFO] Releasing video capture.")
    cap.release()


# --- Flask API Routes ---

@app.route('/health')
def health_check():
    """A simple endpoint to check if the server is running."""
    return "OK", 200

@app.route('/debug_encodings')
def debug_encodings():
    """Debug endpoint to check loaded encodings."""
    try:
        return jsonify({
            "status": "success",
            "encodings_loaded": len(data["encodings"]),
            "names": data["names"],
            "encoding_shapes": [enc.shape for enc in data["encodings"]]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/test_encoding')
def test_encoding():
    """Test endpoint to verify encoding process with dataset images."""
    try:
        import os
        from imutils import paths
        
        results = []
        imagePaths = list(paths.list_images("dataset"))
        
        for imagePath in imagePaths:
            name = imagePath.split(os.path.sep)[-2]
            print(f"[DEBUG] Testing encoding for {name} from {imagePath}")
            
            # Load and encode the image
            image = cv2.imread(imagePath)
            if image is None:
                results.append({"name": name, "status": "error", "message": "Failed to load image"})
                continue
                
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            
            if not boxes:
                results.append({"name": name, "status": "error", "message": "No face found in image"})
                continue
                
            encodings = face_recognition.face_encodings(rgb, boxes)
            if not encodings:
                results.append({"name": name, "status": "error", "message": "Failed to generate encoding"})
                continue
                
            # Test matching with loaded encodings
            encoding = encodings[0]
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
            
            if True in matches:
                match_index = matches.index(True)
                matched_name = data["names"][match_index]
                distances = face_recognition.face_distance(data["encodings"], encoding)
                similarity = 1 - distances[match_index]
                
                results.append({
                    "name": name, 
                    "status": "success", 
                    "matched_with": matched_name,
                    "similarity": round(similarity, 3),
                    "distance": round(distances[match_index], 3)
                })
            else:
                distances = face_recognition.face_distance(data["encodings"], encoding)
                min_distance = min(distances)
                results.append({
                    "name": name, 
                    "status": "failed", 
                    "message": f"No match found. Best distance: {min_distance:.3f}"
                })
        
        return jsonify({"status": "success", "results": results})
        
    except Exception as e:
        print(f"[ERROR] During encoding test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/capture_debug')
def capture_debug():
    """Capture and save current frame for debugging."""
    try:
        stream_url = f"{ESP32_CAM_URL}/capture"
        
        with urllib.request.urlopen(stream_url, timeout=5) as img_resp:
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            
            if frame is None:
                return jsonify({"status": "error", "message": "Failed to capture frame"})
            
            # Save the frame for debugging
            debug_filename = f"debug_frame_{int(time.time())}.jpg"
            cv2.imwrite(debug_filename, frame)
            
            # Try to detect faces in the captured frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            
            # If faces are detected, try to generate encodings
            encodings = []
            if boxes:
                encodings = face_recognition.face_encodings(rgb_frame, boxes)
            
            return jsonify({
                "status": "success",
                "frame_saved": debug_filename,
                "frame_shape": frame.shape,
                "faces_detected": len(boxes),
                "face_locations": boxes,
                "encodings_generated": len(encodings),
                "encoding_shapes": [enc.shape for enc in encodings] if encodings else []
            })
            
    except Exception as e:
        print(f"[ERROR] During debug capture: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/test_attendance_logging/<name>')
def test_attendance_logging(name):
    """Test endpoint to manually test attendance logging without camera."""
    try:
        print(f"[DEBUG] Testing attendance logging for: {name}")
        
        # Check if name exists in student data
        if name not in student_df['Name'].values:
            return jsonify({"status": "error", "message": f"Name '{name}' not found in student data"})
        
        # Test the log_attendance function
        result = log_attendance(name)
        
        # Read the attendance log to show current state
        if os.path.exists(ATTENDANCE_LOG_CSV):
            attendance_log = pd.read_csv(ATTENDANCE_LOG_CSV)
            current_entries = len(attendance_log)
            print(f"[DEBUG] Current attendance log has {current_entries} entries")
            if current_entries > 0:
                print(f"[DEBUG] Latest entries: {attendance_log.tail().to_dict('records')}")
        else:
            current_entries = 0
        
        return jsonify({
            "test_result": result,
            "attendance_log_entries": current_entries,
            "message": f"Test completed for {name}"
        })
        
    except Exception as e:
        print(f"[ERROR] During attendance logging test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/video_feed')
def video_feed():
    """The main video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_face', methods=['GET'])
def detect_face():
    """Endpoint to detect if there's a face in the current frame."""
    stream_url = f"{ESP32_CAM_URL}/capture" 
    try:
        print("[DEBUG] Face detection request received")
        
        with urllib.request.urlopen(stream_url, timeout=5) as img_resp:
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is None:
                print("[DEBUG] Failed to decode frame in face detection")
                return jsonify({"status": "error", "message": "Could not retrieve frame from camera."})

            print(f"[DEBUG] Face detection frame shape: {frame.shape}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            print(f"[DEBUG] Face detection found {len(boxes)} faces")
            
            if boxes:
                return jsonify({"status": "success", "face_detected": True, "face_count": len(boxes)})
            else:
                return jsonify({"status": "success", "face_detected": False, "face_count": 0})

    except Exception as e:
        print(f"[ERROR] During face detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "An internal server error occurred during face detection."})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """Endpoint to trigger face recognition and mark attendance."""
    stream_url = f"{ESP32_CAM_URL}/capture" 
    try:
        print("[DEBUG] Starting face recognition process...")
        
        with urllib.request.urlopen(stream_url, timeout=5) as img_resp:
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is None:
                print("[DEBUG] Failed to decode frame from camera")
                return jsonify({"status": "error", "message": "Could not retrieve frame from camera."})

            print(f"[DEBUG] Frame shape: {frame.shape}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            print(f"[DEBUG] Face locations detected: {len(boxes)}")
            
            if not boxes:
                print("[DEBUG] No face locations found")
                return jsonify({"status": "error", "message": "No face detected."})

            encodings = face_recognition.face_encodings(rgb_frame, boxes)
            print(f"[DEBUG] Face encodings generated: {len(encodings)}")

            if not encodings:
                print("[DEBUG] No face encodings generated")
                return jsonify({"status": "error", "message": "No face detected."})

            encoding = encodings[0]
            print(f"[DEBUG] Using first encoding, shape: {encoding.shape}")
            print(f"[DEBUG] Comparing with {len(data['encodings'])} known encodings")
            
            # Use a more lenient tolerance for face comparison
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.7)
            print(f"[DEBUG] Match results: {matches}")
            
            if True in matches:
                match_index = matches.index(True)
                name = data["names"][match_index]
                print(f"[DEBUG] Face matched with: {name}")
                
                # Calculate similarity score
                distances = face_recognition.face_distance(data["encodings"], encoding)
                similarity_score = 1 - distances[match_index]
                print(f"[DEBUG] Similarity score: {similarity_score:.3f}")
                
                log_result = log_attendance(name)
                return jsonify(log_result)
            else:
                # Calculate distances to see how close we are
                distances = face_recognition.face_distance(data["encodings"], encoding)
                print(f"[DEBUG] Face distances: {distances}")
                min_distance = min(distances)
                print(f"[DEBUG] Minimum distance: {min_distance:.3f} (threshold: 0.6)")
                
                return jsonify({"status": "error", "message": f"Unknown person. Best match distance: {min_distance:.3f}"})

    except Exception as e:
        print(f"[ERROR] During attendance marking: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "An internal server error occurred. Is the camera URL correct?"})

@app.route('/get_attendance')
def get_attendance():
    """Endpoint to retrieve all attendance records."""
    try:
        attendance_df = pd.read_csv(ATTENDANCE_LOG_CSV)
        
        # Only sort if there are rows to sort
        if len(attendance_df) > 0:
            attendance_df = attendance_df.sort_values(by='Timestamp', ascending=False)
        
        return jsonify(attendance_df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        print(f"[ERROR] Could not read attendance log: {e}")
        return jsonify({"error": "Could not retrieve attendance data."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)