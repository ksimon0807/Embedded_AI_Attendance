from imutils import paths
import face_recognition
import pickle
import cv2
import os

print("[INFO] Starting face encoding process...")
imagePaths = list(paths.list_images("dataset"))

knownEncodings = []
knownNames = []
totalFacesEncoded = 0

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
    # Extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]

    # Load the input image and convert it from BGR to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes for each face
    boxes = face_recognition.face_locations(rgb, model="hog")

    # If no faces are found in the image, skip it with a warning
    if not boxes:
        print(f"[WARNING] No faces found in {imagePath}. Skipping this image.")
        continue

    # Compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings (in case there are multiple faces in one image)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        totalFacesEncoded += 1

print("-" * 30)
# Dump the facial encodings + names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

# Ensure the output file is named correctly
output_encoding_file = "encodings.pickle"
with open(output_encoding_file, "wb") as f:
    f.write(pickle.dumps(data))

# --- Final Summary ---
uniqueNames = len(set(knownNames))
print(f"[SUCCESS] Encoding complete.")
print(f"[SUMMARY] Encoded {totalFacesEncoded} faces from {uniqueNames} unique students.")
print(f"[SUMMARY] Saved encodings to '{output_encoding_file}'")
print("-" * 30)