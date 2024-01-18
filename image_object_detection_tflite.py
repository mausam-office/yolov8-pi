"""Evaluates single image at a time"""
import datetime
import time
import cv2
import os
import random
import sys
import numpy as np
import tensorflow as tf

from datetime import datetime
from yolov8.utils import xywh2xyxy, nms, draw_detections

interpreter = None
org_img_shape = None

def load_model(model_path):
    global interpreter
    # load Yolov8 tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

def load_input_image(img_path, input_shape):
    global org_img_shape
    img = cv2.imread(img_path)
    org_img_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[2], input_shape[3])) # CANGES FOR tf AND pt

    # Scale input pixel values to 0 to 1
    img = img / 255.0
    img = img.transpose(2, 0, 1) # CANGES FOR tf AND pt 
    input_tensor = img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def rescale_boxes(boxes, input_shape):
    # dimension of model input 
    input_width = input_shape[3]
    input_height = input_shape[2]

    # dimension of input image
    img_width = org_img_shape[1]
    img_height = org_img_shape[0]

    # Rescale boxes to original image dimensions
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes


def extract_boxes(predictions, input_shape):
    # Extract boxes from predictions
    boxes = predictions[:, :4]
    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, input_shape)
    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)
    return boxes


def process_output(detections, input_shape, conf=None, iou=None):
    #TODO make dynamic
    conf_threshold = conf
    iou_threshold = iou
    # print(f"shape of detections: {detections.shape}")
    predictions = np.squeeze(detections).T
    # print(f"shape of predictions: {predictions.shape}")

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    # print(f"scores: {len(scores)} {scores.shape}",scores)
    predictions = predictions[scores > conf_threshold, :]
    # print(f"predictions: {len(predictions)} {predictions.shape}",predictions)
    scores = scores[scores > conf_threshold]
    # print(f"scores: {len(scores)} {scores.shape}",scores)

    if len(scores) == 0:
        return [], [], []
    
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    # print(f"class_ids: {len(class_ids)} {class_ids.shape}", class_ids)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, input_shape)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)
    return boxes[indices], scores[indices], class_ids[indices]
    

def detect(img_path, conf=None, iou=None):
    if conf is None:
        conf = 0.5
    if iou is None:
        iou = 0.1
    
    # get input and output tensor details of model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print('input_details: ', input_details)
    # print('output_details: ', output_details)

    # input_shape is [batch, channel, height, width] <-> [1, 3, 640, 640]
    input_shape = input_details[0]['shape']

    # get input tensor
    input_tensor = load_input_image(img_path, input_shape)

    # inference 
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    detections = interpreter.get_tensor(output_details[0]['index'])[0]

    # post-processing
    boxes, scores, class_ids = process_output(detections, input_shape, conf=conf, iou=iou)
    return boxes, scores, class_ids, input_shape


def detection(img_path, mode=None, save_output=None, show_output=None, wait=None):
    start = time.time()
    for _ in range(50):
        if mode != 'image': 
            wait = False
        boxes, scores, class_ids, input_shape = detect(img_path, conf=conf_thresh, iou=iou_thresh)

        img = cv2.imread(img_path)

        # img = draw_detections(img, boxes, scores, class_ids)

        if show_output:
            # cv2.namedWindow("Tflite Detected Objects", cv2.WINDOW_NORMAL)
            cv2.imshow("Tflite Detected Objects", img)

        if save_output:
            cv2.imwrite(f"{output_dir}/{str(datetime.now().timestamp())}.jpg", img)

        if wait and show_output:
            cv2.waitKey(9000)
        else:
            cv2.waitKey(1)
    print(f"Average FPS: {50/(time.time()-start)}")


def main(mode=None, save_output=False, show_output=False, num_frame_skip=1):
    start_loading = time.perf_counter()
    load_model(TFLITE_MODEL_PATH)
    print(f"Model Load Duration: {time.perf_counter() - start_loading}")
    frame_path = 'CV_frame.jpg'
    
    if mode is None:
        print('No mode is provided. Choose one out of `image`, `webcam` and `video`')

    if mode == 'webcam':
        cam =cv2.VideoCapture(0)
        while True:
            success, frame = cam.read()
            cv2.imwrite(frame_path, frame)
            detection(frame_path, mode, save_output, show_output)
             
    elif mode == 'image':
        for _ in range(1):
            start_loading = time.perf_counter()
            detection(IMG_PATH, mode, save_output, show_output)
            print(f"Inference Duration: {time.perf_counter() - start_loading}")

    elif mode == 'video':
        cam = cv2.VideoCapture(VIDEO_PATH)
        scale_factor = 1
        num_frame_skip = num_frame_skip
        frame_skip = 0
        counter = 0
        while True:
            success, frame = cam.read()
            if not success:
                break
            if frame_skip < num_frame_skip:
                frame_skip += 1
                print('-'*frame_skip + '>' + f" skipping {frame_skip} time")
                continue
            # frame = cv2.resize(frame, (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor)))
            cv2.imwrite(frame_path, frame)
            detection(frame_path, mode, save_output, show_output)
            counter += 1
            frame_skip = 0
    else:
        pass  

    
if __name__ == "__main__":
    # TFLITE_MODEL_PATH = 'runs/detect/train2/weights/exp2_best_metadata.tflite'
    TFLITE_MODEL_PATH = "pre-trained-models/yolov8n-metadata.tflite"
    IMG_PATH = 'CV_frame.jpg'
    IMG_PATH = 'images/samples/00021_road_birds.jpg'
    # VIDEO_PATH = 'E:/pramod/bird_in_pole/dataset/test/test24.mp4'
    output_dir = "images/output/00021_road_birds.jpg"

    conf_thresh = 0.55
    iou_thresh = 0.05
    mode = 'image'
    save_output = False
    show_output = False
    num_frame_skip = 5
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not TFLITE_MODEL_PATH.endswith('.tflite'):
        print(f"Inappropriate model format.")
        sys.exit()

    if mode=='image' and not any([IMG_PATH.endswith('.jpg'), IMG_PATH.endswith('.jpeg')]):
        print(f"Image type must be with .jpg or .jpeg extension.")
        sys.exit()
        
    if not 0<conf_thresh<1 or not 0<iou_thresh<1:
        print(f"Threshold values must be between `0` and `1`")
        sys.exit()

    # Printing all values of array without truncation
    np.set_printoptions(threshold=sys.maxsize)

    # Choose one out of `image`, `webcam` and `video for `mode`
    main(mode, save_output, show_output, num_frame_skip)
