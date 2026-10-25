from ultralytics import YOLO
import cv2

model = YOLO('yolov13s.pt')

image_path = 'test.jpg'

results = model(image_path)

results[0].save('output.jpg')

print(f"✅ {len(results[0].boxes)} شیء در تصویر شناسایی شد!")

print("\n📋 لیست اشیاء شناسایی شده:")
for box in results[0].boxes:
    class_name = model.names[int(box.cls)]
    confidence = box.conf[0]
    print(f"   - {class_name}: {confidence:.2f}%")

print("\n📍 مختصات جعبه‌ها (x1, y1, x2, y2):")
for box in results[0].boxes:
    print(f"   {box.xyxy[0].tolist()}")