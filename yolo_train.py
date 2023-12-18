from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='targetDetect.yaml', epochs=5)

model.val()