import pickle
import os
from imutils import paths
import cv2
import face_recognition

print("=== CHECKING ENCODINGS FILE ===")

# Check if encodings file exists
if os.path.exists("encodings.pickle"):
    print("✓ encodings.pickle file found")
    
    # Load encodings
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
        
        print(f"✓ Encodings loaded successfully")
        print(f"  - Number of encodings: {len(data['encodings'])}")
        print(f"  - Names: {data['names']}")
        print(f"  - Encoding shapes: {[enc.shape for enc in data['encodings']]}")
        
        # Check if encodings are valid
        for i, encoding in enumerate(data['encodings']):
            if encoding is not None and len(encoding) > 0:
                print(f"  ✓ Encoding {i} ({data['names'][i]}): Valid, shape {encoding.shape}")
            else:
                print(f"  ✗ Encoding {i} ({data['names'][i]}): INVALID or empty")
                
    except Exception as e:
        print(f"✗ Error loading encodings: {e}")
else:
    print("✗ encodings.pickle file NOT found")

print("\n=== CHECKING DATASET IMAGES ===")

# Check dataset directory
if os.path.exists("dataset"):
    print("✓ dataset directory found")
    
    # List all images
    imagePaths = list(paths.list_images("dataset"))
    print(f"  - Total images found: {len(imagePaths)}")
    
    for imagePath in imagePaths:
        name = imagePath.split(os.path.sep)[-2]
        print(f"  - {name}: {imagePath}")
        
        # Try to load and encode the image
        try:
            image = cv2.imread(imagePath)
            if image is None:
                print(f"    ✗ Failed to load image")
                continue
                
            print(f"    ✓ Image loaded, shape: {image.shape}")
            
            # Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                print(f"    ✗ No faces detected in image")
                continue
                
            print(f"    ✓ {len(boxes)} face(s) detected")
            
            # Generate encodings
            encodings = face_recognition.face_encodings(rgb, boxes)
            if not encodings:
                print(f"    ✗ Failed to generate encodings")
                continue
                
            print(f"    ✓ {len(encodings)} encoding(s) generated, shape: {encodings[0].shape}")
            
        except Exception as e:
            print(f"    ✗ Error processing image: {e}")
else:
    print("✗ dataset directory NOT found")

print("\n=== SUMMARY ===")
if os.path.exists("encodings.pickle") and os.path.exists("dataset"):
    print("Both encodings file and dataset exist. Check the details above for any issues.")
else:
    print("Missing files detected. Please run encode_faces.py first.")
