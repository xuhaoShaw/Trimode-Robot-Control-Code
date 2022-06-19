import os
import time
import threading
import numpy as np
import pyrealsense2 as rs
from Vision import Vision
from queue import Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def put_frame():
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data(), dtype="float16")

        q.put(depth_data)

def get_frame():
    while True:
        depth = q.get()
        if depth is None:
            continue
        print('Processing frame...')
        time_start = time.time()
        my_vision.process_frame(depth)
        print(f'Processed frame in {time.time()-time_start} s.')
        time.sleep(0.2)

if __name__ == "__main__":
    #  Configure realsense
    pipeline = rs.pipeline()  # 管道配置文件
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)

    lock = threading.Lock()
    q = Queue(maxsize=1)
    my_vision = Vision()
    PUT_FRAME = threading.Thread(target=put_frame)
    GET_FRAME = threading.Thread(target=get_frame)

    PUT_FRAME.start()
    GET_FRAME.start()
    PUT_FRAME.join()
    GET_FRAME.join()

