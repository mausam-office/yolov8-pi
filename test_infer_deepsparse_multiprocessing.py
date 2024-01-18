import asyncio
from datetime import datetime
import os
import cv2
import itertools
import numpy as np
import time

from deepsparse import Pipeline
from multiprocessing import Process, Queue
from threading import Thread

INPUT_Q_SIZE = 5

img_q1 = Queue(INPUT_Q_SIZE)
img_q2 = Queue(INPUT_Q_SIZE)
# img_q3 = Queue()
output_q = Queue()

queues = [img_q1, img_q2]

queue_iterator = itertools.cycle(queues)

cam = cv2.VideoCapture('/mnt/d/Mausam/YOLOv8/Vehicle detection/datasets/test-data/car.mp4')

def inference_process(img_q: Queue, output_q: Queue, p_name:str):
    model_path = './exported/deployment/model.onnx'
    yolo_pipeline = Pipeline.create(
        task="yolov8",
        model_path=model_path,
    )

    while True:
        if not img_q.empty():
            img = img_q.get()
            # print(f"{img.shape=}")
            # filename = f"data/{p_name}_file.jpg"
            # cv2.imwrite(filename, img)
            try:
                pipeline_outputs = yolo_pipeline(images=[img])
                output_q.put(pipeline_outputs)
                # print(f"{pipeline_outputs=}")
                print(f'{p_name} output at {datetime.now()}')
            except Exception as e:
                print(f"Error {e}")
            # os.remove(filename)

                
def get_images():
    input_dir = './images/samples'
    images = os.listdir(input_dir)
    images = [os.path.join(input_dir, img_name) for img_name in images]
    for image in images:
        yield image

def cam_images(cam):
    # print('cam func')
    
    # if cam.isOpened():
    ret, frame = cam.read()
    # print(f"{type(frame) = }")
    if not ret:
        print("not frame")
    else:
        return frame

def insert_data(queue_iterator, INPUT_Q_SIZE):
    cond = True
    count = 3
    while cond:
        # imgs = get_images()
        # for img in imgs:
        #     next(queue_iterator).put(img)
        current_q: Queue = next(queue_iterator)
        img = cam_images(cam)
        if img is None or current_q.qsize()>=INPUT_Q_SIZE:
            continue
        current_q.put(img)
        
        # count -= 1 
        # if count < 0:
        #     cond = False
        #     print(f"condition is false")
        # time.sleep(0.2)

def post_processing(output_q):
    while True:
        if not output_q.empty():
            outputs = output_q.get()
            # print()
        

def main():
    infer_process1 = Process(target=inference_process, args=(img_q1, output_q, 'p1'))
    infer_process1.start()

    infer_process2 = Process(target=inference_process, args=(img_q2, output_q, 'p2'))
    infer_process2.start()

    thread1_input = Thread(target=insert_data, args=(queue_iterator, INPUT_Q_SIZE))
    thread1_input.start()
    thread2_input = Thread(target=post_processing, args=(output_q,))
    thread2_input.start()

    
    # cond = True
    # count = 3
    
    # while cond:
    #     # imgs = get_images()
    #     # for img in imgs:
    #     #     next(queue_iterator).put(img)
    #     img = cam_images(cam)
    #     if img is None:
    #         continue
    #     next(queue_iterator).put(img)

    #     if not output_q.empty():
    #         print(output_q.get())
        
    #     count -= 1 
    #     if count < 0:
    #         cond = False
    #         print(f"condition is false")
    #     time.sleep(3)
    infer_process1.join()
    infer_process2.join()
    thread1_input.join()
    thread2_input.join()

    
if __name__ == "__main__":
    main()
