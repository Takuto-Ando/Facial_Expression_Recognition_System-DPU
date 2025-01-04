#!/bin/sh

from pynq_dpu import DpuOverlay
from typing import List
import os
import numpy as np
import cv2
import xir
import threading
import queue
import time
import vart
# from vaitrace_py import vai_tracepoint
from pynq.lib.video import *

# Constants and global variables
MAX_FRAMES = 500
FRAME_QUEUE_SIZE = 50
THREAD_COUNT = 2
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
PIXEL_FORMAT = 24
DETECTION_THRESHOLD = 0.5
NMS_THRESHOLD = 0.1
FACE_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Global counters
frame_counter = 0
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
stop_event = threading.Event()

# Load FPGA overlay
overlay = DpuOverlay("./bit/dpu_b512.bit")


# Softmax computation
def softmax(data):
    exp_data = np.exp(data)
    sum_exp_data = np.sum(exp_data, axis=1, keepdims=True)
    return exp_data / sum_exp_data


# Non-Maximum Suppression (NMS)
def nms_boxes(boxes, scores, nms_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(overlap <= nms_threshold)[0]
        order = order[inds + 1]

    return keep


# Run the DPU for face detection and emotion recognition
@vai_tracepoint
def run_dpu(dpu_fd, dpu_fer):
    global frame_counter

    while not stop_event.is_set() and frame_counter < MAX_FRAMES:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        faces, scores = run_face_detection(dpu_fd, frame)
        recognized_faces = run_emotion_recognition(dpu_fer, frame, faces)

        frame_counter += 1
        if frame_counter >= MAX_FRAMES:
            stop_event.set()


def run_face_detection(dpu, frame):
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()

    input_shape = tuple(input_tensors[0].dims)
    output_shape = tuple(output_tensors[0].dims)

    input_data = np.empty(input_shape, dtype=np.float32, order='C')
    output_data = [
        np.empty(output_shape, dtype=np.float32, order='C'),
        np.empty(output_shape, dtype=np.float32, order='C'),
    ]

    # Preprocess input
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = img - 128.0
    img = np.expand_dims(img, axis=0)
    input_data[0] = img

    # Execute DPU
    job_id = dpu.execute_async([input_data], output_data)
    dpu.wait(job_id)

    # Extract and process outputs
    bboxes = np.reshape(output_data[0], (-1, 4))
    scores = np.reshape(output_data[1], (-1, 2))
    probabilities = softmax(scores)[:, 1]

    # Filter by detection threshold
    valid_indices = probabilities > DETECTION_THRESHOLD
    bboxes = bboxes[valid_indices]
    probabilities = probabilities[valid_indices]

    # Apply Non-Maximum Suppression
    nms_indices = nms_boxes(bboxes, probabilities, NMS_THRESHOLD)
    return bboxes[nms_indices], probabilities[nms_indices]


def run_emotion_recognition(dpu, frame, faces):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_size = (48, 48)
    recognized_faces = []

    for face in faces:
        xmin, ymin, xmax, ymax = face
        roi_gray = gray_frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        face_image = cv2.resize(roi_gray, target_size, interpolation=cv2.INTER_AREA)
        face_image = face_image.reshape(48, 48, 1) * (1 / 255.0) * 64

        input_tensors = dpu.get_input_tensors()
        input_data = np.empty(tuple(input_tensors[0].dims), dtype=np.int8, order='C')
        input_data[0] = face_image

        output_tensors = dpu.get_output_tensors()
        output_data = [np.empty(tuple(output_tensors[0].dims), dtype=np.int8, order='C')]

        # Execute DPU
        job_id = dpu.execute_async([input_data], output_data)
        dpu.wait(job_id)

        # Extract emotion class
        emotion_scores = output_data[0].reshape(1, -1)
        emotion_class = FACE_CLASSES[np.argmax(emotion_scores)]
        recognized_faces.append((face, emotion_class))

    return recognized_faces


# Frame capture thread
def capture_frames(video_capture):
    global frame_counter
    while not stop_event.is_set() and frame_counter < MAX_FRAMES:
        ret, frame = video_capture.read()
        if not ret:
            continue
        try:
            frame_queue.put(frame, timeout=0.001)
        except queue.Full:
            continue


# Get DPU subgraphs
def get_dpu_subgraphs(graph_path):
    graph = xir.Graph.deserialize(graph_path)
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()

    return [
        subgraph for subgraph in child_subgraphs
        if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU"
    ]


# Main function
def main():
    global frame_counter

    # Load DPU models
    fd_subgraphs = get_dpu_subgraphs("model/dense_b512.xmodel")
    fer_subgraphs = get_dpu_subgraphs("model/cnn705_b512.xmodel")

    fd_runners = [vart.Runner.create_runner(fd_subgraphs[0], "run") for _ in range(THREAD_COUNT)]
    fer_runners = [vart.Runner.create_runner(fer_subgraphs[0], "run") for _ in range(THREAD_COUNT)]

    # Initialize video capture and display
    video_capture = cv2.VideoCapture(0 + cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    print("Capture device is open: " + str(video_capture.isOpened()))

    # Start threads
    threads = []
    capture_thread = threading.Thread(target=capture_frames, args=(video_capture,))
    capture_thread.start()
    threads.append(capture_thread)

    for i in range(THREAD_COUNT):
        thread = threading.Thread(target=run_dpu, args=(fd_runners[i], fer_runners[i]))
        thread.start()
        threads.append(thread)

    # Wait for threads to finish
    try:
        while frame_counter < MAX_FRAMES:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    for thread in threads:
        thread.join()




if __name__ == "__main__":
    main()