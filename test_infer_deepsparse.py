
import os
import cv2
import time
import numpy as np
from deepsparse import Pipeline

input_dir = './images/samples'
output_dir = './images/output'
os.makedirs(output_dir, exist_ok=True)

model_path = './exported/deployment/model.onnx'

yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path=model_path,
)

# cam = cv2.VideoCapture('E:/pramod/bird_in_pole/dataset/test/test2_Trim.mp4')

# images = [f'{input_dir}/00021_road_birds.jpg', f'{input_dir}/000128_bird_small.jpg']
images = os.listdir(input_dir)
images = [os.path.join(input_dir, img_name) for img_name in images]

fps = 0
iter_num = len(images)
total_duration = 0

for idx in range(iter_num):
    begin = time.perf_counter()

    image = cv2.imread(images[idx])
    
    pipeline_outputs = yolo_pipeline(images=[image])
    
    # print(pipeline_outputs.labels[0])
    # print(pipeline_outputs.scores[0])
    # print(pipeline_outputs.boxes[0])

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
    duration = (time.perf_counter()-begin)
    fps = 1 / duration
    print(f'iter {idx} fps {fps}')
    total_duration += duration

print(f"Average FPS: {iter_num/total_duration}")
