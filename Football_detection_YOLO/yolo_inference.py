from ultralytics import YOLO
model_path = r"C:\Users\91823\Downloads\archive\footballDetection\models\best (4).pt"
model  = YOLO(model_path)  # Load a pretrained YOLOv8 model
results = model.predict('input_videos/08fd33_4.mp4',save=True)  # Perform inference on a video file

print('==================')
