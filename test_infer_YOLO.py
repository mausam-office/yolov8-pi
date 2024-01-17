import os
import time
import cv2
from ultralytics import YOLO

input_dir = './images/samples'
output_dir = './images/output'
os.makedirs(output_dir, exist_ok=True)

model = YOLO('./pre-trained-models/yolov8n.pt')

# images = ['./images/00021_road_birds.jpg', './images/000128_bird_small.jpg']
images = os.listdir(input_dir)

fps = 0
iter_num = 20
total_duration = 0

for idx in range(iter_num):
    begin = time.perf_counter()

    image = cv2.imread(images[idx])
    model.predict(image, device='CPU')

    cv2.imwrite('./images_output/yolo_result.jpg', image)
    duration = (time.perf_counter()-begin)
    fps = 1 / duration
    print(f'iter {idx} fps {fps}')
    total_duration += duration

print(f"Average FPS: {iter_num/total_duration}")