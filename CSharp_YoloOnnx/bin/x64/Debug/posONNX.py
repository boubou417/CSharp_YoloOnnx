from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.export(
    format="onnx",
    opset=12,
    dynamic=False,
    simplify=True
)
print('Complete')
