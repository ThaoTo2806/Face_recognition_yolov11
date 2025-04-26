from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv11 model
model = YOLO('best_model.pt')  # Đảm bảo đường dẫn tới mô hình đã được tải xuống

# Đọc hình ảnh mới để kiểm tra
image_path = 'img15.jpg'  # Đảm bảo rằng đường dẫn tới hình ảnh là chính xác

# Đọc ảnh bằng OpenCV
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB (OpenCV mặc định là BGR)

# Thực hiện dự đoán
results = model(img)

# Nếu results là một list, hãy lấy đối tượng đầu tiên trong danh sách
result = results[0]

# Hiển thị kết quả
result.show()  # Hiển thị kết quả ngay trong Colab

# Hoặc nếu bạn muốn hiển thị ảnh bằng matplotlib
plt.imshow(img)
plt.axis('off')  # Ẩn trục
plt.show()

# Lưu lại ảnh với kết quả dự đoán
result.save()  # Lưu hình ảnh với kết quả được vẽ lên
