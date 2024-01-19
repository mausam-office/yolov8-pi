
import os
import cv2
import time
import numpy as np
from deepsparse import Pipeline

input_dir = './images/samples'
output_dir = './images/output'
os.makedirs(output_dir, exist_ok=True)

model_path = './exported/deployment/model.onnx'

cam = cv2.VideoCapture('/mnt/d/Mausam/YOLOv8/Vehicle detection/datasets/test-data/car.mp4')

yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path=model_path,
)

total_duration = 0 
cond = True
count = 0
while cond:
    begin = time.perf_counter()
    ret, image = cam.read()
    if not ret:
        cond = False
        continue

    pipeline_outputs = yolo_pipeline(images=[image])

    boxes = list(zip(pipeline_outputs.boxes[0], pipeline_outputs.labels[0], pipeline_outputs.scores[0]))

    for box, label, score in boxes:
        x1, y1, x2, y2 = box = np.array(box).astype('int')
        # w = int(w/2)
        # h = int(h/2)
        label = int(float(label))
        image = cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2), 
                (255,0,0),
                2
            )
        # print([box, label, score])

    cv2.imwrite('./images/output/deepsparse_result.jpg', image)

    count += 1
    duration = (time.perf_counter()-begin)
    fps = 1 / duration
    print(f'iter fps: {fps}')
    total_duration += duration
    
print(f"Avg fps {count/total_duration}")





