import time
import numpy
import cv2

from pathlib import Path
import onnxruntime_extensions


def display_image(outputs):
    label_map = {
        0 : 'bird',
        1 : 'person',
        2 : 'head',
        3 : 'plane'
    }
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (0, 0), fx = 0.8, fy = 0.8)
    for output in outputs:
        box, (score, label_idx) = output[:4], output[4:]
        x, y, w, h = box.astype('int')
        label_idx = int(label_idx)
        print(x, y, w, h, f"{score*100:.2f}%", label_idx)
        w = int(w/2)
        h = int(h/2)
        image = cv2.rectangle(
            image,
            (x-w, y-h),
            (x+w, y+h), 
            (255,0,0),
            2
        )
    
        # cv2.putText(image, label_map[label_idx], (x-w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('output', image)
    # cv2.waitKey(0)
    # cv2.imwrite('./images/result2.jpg', image)


def test_inference(onnx_model_file:Path):
    import onnxruntime as ort
    import numpy as np

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    
    image = np.frombuffer(open(image_path, 'rb').read(), dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    start = time.perf_counter()
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    # print("here")
    # outputs = session.run(['image_out'], inp)[0]
    # outputs = session.run(['nms_out'], inp)[0]
    outputs = session.run(['scaled_box_out'], inp)[0]
    # open('./images/result.jpg', 'wb').write(outputs)
    # print(outputs, '\n', type(outputs))
    # outname = [i.name for i in session.get_outputs()]
    # print(outname)
    display_image(outputs)
    print(f"Duration {time.perf_counter() - start} seconds")

    start = time.perf_counter()
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    # print("here")
    # outputs = session.run(['image_out'], inp)[0]
    # outputs = session.run(['nms_out'], inp)[0]
    outputs = session.run(['scaled_box_out'], inp)[0]
    # open('./images/result.jpg', 'wb').write(outputs)
    # print(outputs, '\n', type(outputs))
    # outname = [i.name for i in session.get_outputs()]
    # print(outname)
    # display_image(outputs)
    print(f"Duration {time.perf_counter() - start} seconds")
    

if __name__ == '__main__':
    image_path = './images/samples/000128_bird_small.jpg'
    onnx_e2e_model_name = Path("models/yolov8n.with_pre_post_processing.onnx")
    test_inference(onnx_e2e_model_name)