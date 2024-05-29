import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import time

# Initialize video capture
cap = cv2.VideoCapture("IndiraNagar.mp4")

# Initialize YOLO model
model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = {
    "lane1": ([935, 90, 1275, 90], [935, 110, 1275, 110]),
    "lane2": ([1365, 120, 1365, 360], [1385, 120, 1385, 360]),
    "lane3": ([600, 70, 600, 170], [620, 70, 620, 170]),
    "lane4": ([450, 500, 1240, 500], [450, 520, 1240, 520])
}

totalCounts = {
    "lane1": [],
    "lane2": [],
    "lane3": [],
    "lane4": []
}

# Streamlit app
st.title("Traffic Flow Detection")

# Create a placeholder for the video frames
frame_placeholder = st.empty()

def process_frame():
    success, img = cap.read()
    if not success:
        return None

    imgRegion = cv2.bitwise_and(img, img)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for lane, (limit1, limit2) in limits.items():
        cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (250, 182, 122), 2)
        cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (250, 182, 122), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(25, y1)), scale=1, thickness=1, colorR=(56, 245, 213), colorT=(25, 26, 25), offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

        for lane, (limit1, limit2) in limits.items():
            if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
                if totalCounts[lane].count(id) == 0:
                    totalCounts[lane].append(id)
                    cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (12, 202, 245), 3)

            if limit2[0] < cx < limit2[2] and limit2[1] - 15 < cy < limit2[1] + 15:
                if totalCounts[lane].count(id) == 0:
                    totalCounts[lane].append(id)
                    cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (12, 202, 245), 3)

    for lane, totalCount in totalCounts.items():
        lane_num = int(lane[-1])
        cvzone.putTextRect(img, f' {lane_num}st Lane: {len(totalCount)}', (25, 75 + (70 * (lane_num - 1))), 2, thickness=2, colorR=(147, 245, 186), colorT=(15, 15, 15))

    return img

while cap.isOpened():
    frame = process_frame()
    if frame is not None:
        frame_placeholder.image(frame, channels="BGR")
    else:
        st.write("End of video or video not found.")
        break
    time.sleep(0.1)  # Adjust to control the speed of the video

cap.release()
