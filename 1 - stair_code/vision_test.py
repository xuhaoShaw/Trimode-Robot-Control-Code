import os
import time
import threading
import numpy as np
import cv2 as cv
import pyrealsense2 as rs
from Vision import Vision
from queue import Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def put_frame():
    color_path = r'videos/1/color_4.png'
    depth_path = r'videos/1/depth_4.npy'
    color = cv.imread(color_path)
    depth = np.load(depth_path).astype(np.uint16)
    q.put([color, depth])

def get_frame():
    while True:
        data = q.get()
        color = data[0]
        depth = data[1]
        if depth is None:
            continue
        print('Processing frame...')
        time_start = time.time()
        my_vision.process_frame(color, depth)
        print(f'Processed frame in {time.time()-time_start} s.')
        time.sleep(0.2)

if __name__ == "__main__":
    lock = threading.Lock()
    q = Queue(maxsize=1)
    my_vision = Vision()
    PUT_FRAME = threading.Thread(target=put_frame)
    GET_FRAME = threading.Thread(target=get_frame)

    PUT_FRAME.start()
    GET_FRAME.start()
    PUT_FRAME.join()
    GET_FRAME.join()

