import os
import base64
import cv2
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
FACES_FOLDER = "faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load reference database
face_db = {}
for person_name in os.listdir(FACES_FOLDER):
    person_path = os.path.join(FACES_FOLDER, person_name)
    if os.path.isdir(person_path):
        images = [
            os.path.join(person_path, img)
            for img in os.listdir(person_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if images:
            face_db[person_name] = images

@app.route("/")
def index():
    return render_template("index.html")

def save_base64_image(base64_str, save_path):
    # base64_str is like: "data:image/png;base64,...."
    img_data = base64_str.split(",")[1]
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(img_data))

@app.route("/upload", methods=["POST"])
def upload():
    file_path = None

    if "image" in request.files:
        # Uploaded file from input
        file = request.files["image"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

    elif "webcam_image" in request.form:
        # Image from webcam
        file_path = os.path.join(UPLOAD_FOLDER, "webcam_capture.jpg")
        save_base64_image(request.form["webcam_image"], file_path)

    else:
        return jsonify({"error": "No image provided"}), 400

    best_person = None
    best_score = float("inf")

    try:
        for person, images in face_db.items():
            for ref_img in images:
                try:
                    result = DeepFace.verify(
                        img1_path=file_path,
                        img2_path=ref_img,
                        enforce_detection=False
                    )
                    if result["distance"] < best_score:
                        best_score = result["distance"]
                        best_person = person
                except Exception as e:
                    print(f"Skipping {ref_img} due to error: {e}")
                    continue

        # Confidence score
        confidence = round((1 - best_score) * 100, 2)

        # Draw bounding boxes
        image = cv2.imread(file_path)
        detections = DeepFace.extract_faces(img_path=file_path, enforce_detection=False)
        for face in detections:
            fa = face["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        boxed_path = os.path.join(UPLOAD_FOLDER, "boxed_" + os.path.basename(file_path))
        cv2.imwrite(boxed_path, image)

        return jsonify({
            "match": best_person,
            "confidence": confidence,
            "image_path": boxed_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
