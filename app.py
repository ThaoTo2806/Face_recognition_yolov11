from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO('best_model.pt')

# Mảng thông tin người để so sánh
people_data = [
    {"name": "Cao Tấn Công", "age": 22, "GT": "Nam", "class_name": "cong_caos"},
    {"name": "Lê Hữu Tài", "age": 22, "GT": "Nam", "class_name": "tai_le"}
]

@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem có tệp được gửi lên không
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Nếu không có tệp được chọn
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Đọc ảnh và thực hiện dự đoán
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Thực hiện dự đoán
    results = model(img)

    # Lấy kết quả nhận diện đầu tiên
    result = results[0]
    #print(f"Result: {result}")

    # Kiểm tra nếu có đối tượng phát hiện và có xác suất
    if result.probs is not None and len(result.probs) > 0:
        # Lấy tên lớp có xác suất cao nhất
        class_name = result.names[result.probs.argmax().item()]
        print(f"Detected class: {class_name}")  # In ra tên lớp nhận diện được

        # Chuẩn hóa tên lớp (chẳng hạn loại bỏ dấu cách, chuyển thành chữ thường nếu cần)
        class_name = class_name.strip().lower()  # Chuyển thành chữ thường và loại bỏ dấu cách nếu có
        
        # Tìm thông tin của người trong mảng, kiểm tra so sánh class_name
        person_info = next((person for person in people_data if person["class_name"].strip().lower() == class_name), None)
        
        if person_info:
            return jsonify({"message": "Prediction complete", "class_name": class_name, "info": person_info})
        else:
            return jsonify({"message": "Person not found in data."})
    else:
        return jsonify({"message": "No objects detected."})

if __name__ == '__main__':
    app.run(debug=True)
