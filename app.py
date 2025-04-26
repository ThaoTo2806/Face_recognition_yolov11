from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO('best_model.pt')

# Mảng thông tin người để so sánh
people_data = [
    {"name": "Cao Tấn Công", "age": 22, "GT": "Nam", "class_name": "cong_cao"},
    {"name": "Lê Hữu Tài", "age": 22, "GT": "Nam", "class_name": "tai_le"}
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img)
    result = results[0]

    # Lấy tất cả tên lớp phát hiện
    detected_classes = [result.names[int(cls)] for cls in result.boxes.cls]
    print(f"Detected classes: {detected_classes}")

    # Dùng set để loại trùng
    unique_classes = set([cls.strip().lower() for cls in detected_classes])

    # Tìm người tương ứng
    people_detected = []
    for class_name in unique_classes:
        person_info = next((person for person in people_data if person["class_name"].strip().lower() == class_name), None)
        if person_info:
            people_detected.append({
                "name": person_info["name"],
                "age": person_info["age"],
                "class_name": class_name
            })

    if people_detected:
        return jsonify({"message": "Prediction complete", "people_detected": people_detected})
    else:
        return jsonify({"message": "No matching person found."})

if __name__ == '__main__':
    app.run(debug=True)
