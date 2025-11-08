# -*- coding: utf-8 -*-
"""
Created on Sun Mar 07 19:48:35 2024

@author: MaxGr
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime

current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

import torch
print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")

import yaml

file_path = "OPENAI_API_KEY.yaml"
with open(file_path, "r") as f:
    data = yaml.safe_load(f)
    OPENAI_API_KEY = data["OPENAI_API_KEY"]

import os
import time
import threading
from func_timeout import func_set_timeout

import openai
from openai import OpenAI

client = OpenAI(
  api_key=OPENAI_API_KEY,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_sensitivity = 'normal'

prompt_background = "{You are an voice assistant for blind person, \
the input is the actual data collected by a phone camera, the phone is always facing front, \
please provide the key information for the blind user to help him navigate and avoid potential danger. \
Please note that the xloc and yloc represent the object location (proportional to the image), \
object height and width are also a proportion.}"

prompt_location = "{The location information (center_x, center_y, height, width) of objects is the proportion to the image, \
the detected objects are categorized into 4 type based on the image region.\
Left and Right: objects located on left 25% or right 25% of the image, these objects are usually moving and has large proportion.\
Front: objects that are still far away, can be used to discriminate the current situation.\
Ground: objects that are nearby, need to be cautioned.}"

prompt_motion = "{Analyze the movement (speed and direction) \
and location (xloc and yloc) of each object to determine its trajectory relative to the user.\
Use this information to assess whether an object is moving towards the user and if so, \
how quickly a potential collision might occur based on the object's speed and direction of movement.}"

prompt_sensitivity = '{System sensitivity: Incorporate the sensitivity setting in your response. \
For a low sensitivity setting, identify and report only imminent and direct threats to safety. \
For normal sensitivity, include potential hazards that could pose a risk if not avoided. \
For high sensitivity, report all detected objects that could potentially cause any level of inconvenience or danger.\
More focus on pedestrians and less focus on cars, as users are mostly walking on the sidewalk. \
Please more focus on the left,right,and ground area, as they are usually very close,\
but when you evaluate the emergency, consider the size and type of objects.\
Current sensitivity: ' + str(system_sensitivity) + '}'

instruction = prompt_background + prompt_location + prompt_sensitivity

@func_set_timeout(10)
def GPT_response(model, prompt):
    completion = client.chat.completions.create(
        model=model,
      messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
      ]
    )
    return completion

prompt_format_benchmark = 'Please organize your output into this format: \
{ "danger_score": output 1 for immediate threat, output 0 if not; \
  "reason": explain your annotation reason within 10 words.}'

prompt_word_limiter = 'Limit your answer into 20 words'

GPT_list = []
gpt_request_count = 0
def GPT_annotation(frame_info_i):
    global GPT_list
    object_info = str(frame_info_i)
    prompt = object_info + prompt_format_benchmark

    GPT_start_time = time.time()
    try:
        completion = GPT_response("gemini-2.0-flash", prompt)
        response = completion.choices[0].message.content
        usage = completion.usage
    except:
        print('GPT time running out...')
        return
    GPT_end_time = time.time()
    GPT_time_cost = round(GPT_end_time-GPT_start_time, 4)

    GPT_list.append([response, GPT_time_cost, usage])
    print(response)

from typing import List
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch

import supervision as sv
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()

from ultralytics.trackers.byte_tracker import BYTETracker, STrack 
from ultralytics import YOLOWorld, YOLO

weight_file = 'yolov8x-worldv2.pt'
model = YOLO(weight_file)

# ---------- 偵測類別（維持原本設定） ----------
VIN = [
    'car', 'person', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'stop sign',
    'fountain','crosswalk', 'sidewalk', 'door', 'stair', 'escalator', 'elevator', 'ramp',
    'bench', 'trash can', 'pole', 'fence', 'tree', 'dog', 'cat', 'bird', 'parking meter',
    'mailbox', 'manhole', 'puddle', 'construction sign', 'construction barrier',
    'scaffolding', 'hole', 'crack', 'speed bump', 'curb', 'guardrail', 'traffic cone',
    'traffic barrel', 'pedestrian signal', 'street sign', 'fire hydrant', 'lamp post',
    'bench', 'picnic table', 'public restroom', 'fountain', 'statue', 'monument',
    'directional sign', 'information sign', 'map', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign',
    'stairs sign', 'escalator sign', 'elevator sign', 'restroom sign', 'men restroom sign',
    'women restroom sign', 'unisex restroom sign', 'baby changing station',
    'wheelchair accessible sign', 'braille sign', 'audio signal device', 'tactile paving',
    'detectable warning surface', 'guide rail', 'handrail', 'turnstile', 'gate',
    'ticket barrier', 'security checkpoint', 'metal detector', 'baggage claim',
    'lost and found', 'information desk', 'meeting point', 'waiting area', 'seating area',
    'boarding area', 'disembarking area', 'charging station', 'water dispenser',
    'vending machine', 'ATM', 'kiosk', 'public telephone', 'public Wi-Fi hotspot',
    'emergency phone', 'first aid station', 'defibrillator','tree', 'pole', 'lamp post', 'staff', 'road hazard'
]

URBAN_WALKING_HAZARDS = [
    'person', 'cyclist', 'car', 'bus', 'motorcycle', 'scooter', 'fountain', 'bench', 
    'traffic light', 'stop sign', 'curb', 'ramp', 'stair', 'escalator','charging station',
    'elevator', 'trash can', 'pole', 'tree', 'fire hydrant', 'lamp post','ATM', 'kiosk',
    'construction barrier', 'construction sign', 'scaffolding', 'hole', 'crack', 'speed bump',
    'puddle', 'manhole', 'drain', 'grate', 'loose gravel', 'ice patch', 'snow pile', 'leaf pile',
    'standing water', 'mud', 'sand', 'street sign', 'directional sign', 'information sign',
    'parking meter', 'mailbox', 'bicycle rack', 'outdoor seating', 'planter box', 'bollard',
    'guardrail', 'traffic cone', 'traffic barrel', 'pedestrian signal', 'crowd', 'animal', 'dog',
    'bird', 'cat', 'public restroom', 'fountain', 'statue', 'monument', 'picnic table',
    'outdoor advertisement', 'vendor cart', 'food truck', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign', 'stairs sign',
    'escalator sign', 'elevator sign', 'restroom sign', 'braille sign', 'audio signal device',
    'tactile paving', 'detectable warning surface', 'guide rail', 'handrail', 'turnstile',
    'gate', 'security checkpoint', 'water dispenser', 'vending machine', 
    'public telephone', 'emergency phone', 'first aid station', 'defibrillator',
    'recently paved asphalt', 'oil spill', 'road debris', 'overhanging branches',
    'low-hanging signage', 'temporary road signs', 'roadworks', 'excavation sites', 'utility works',
    'fallen objects', 'spilled cargo', 'flood', 'ice', 'snowdrift', 'landslide debris',
    'erosion damage', 'parked vehicles', 'moving equipment',
    'street performers', 'demonstrations', 'large gatherings', 'parade', 'marathon', 'street fair',
    'temporary scaffolding','electrical hazards', 'wire tangle', 'unsecured manhole covers'
]

CLASSES = URBAN_WALKING_HAZARDS
model.set_classes(CLASSES)

# ---------- utils (未更動) ----------
def calculate_movements(data_previous, tracker_id_previous, data_current, tracker_id_current):
    movements = {}
    bbox_map_previous = {tid: bbox for tid, bbox in zip(tracker_id_previous, data_previous) if tid is not None}
    bbox_map_current = {tid: bbox for tid, bbox in zip(tracker_id_current, data_current) if tid is not None}
    for tid_previous, bbox_previous in bbox_map_previous.items():
        if tid_previous in bbox_map_current:
            center_previous = ((bbox_previous[0] + bbox_previous[2]) / 2, (bbox_previous[1] + bbox_previous[3]) / 2)
            bbox_current = bbox_map_current[tid_previous]
            center_current = ((bbox_current[0] + bbox_current[2]) / 2, (bbox_current[1] + bbox_current[3]) / 2)
            dx = center_current[0] - center_previous[0]
            dy = center_current[1] - center_previous[1]
            movements[tid_previous] = (dx, dy)
    return movements

# ---------- Video setting ----------
text_color = [(0, 255, 0),(0, 0, 255)]
mark_danger = False

fps = 5
display_start_frame = 0
display_until_frame = 10000

video_path = './Video/'
video_name = '1106.mp4'

video_capture = cv2.VideoCapture(video_path + video_name)  
if not video_capture.isOpened():
    print("Error opening video file.")
    exit()

for i in range(1):
    ret, frame = video_capture.read()

img_height, img_width = frame.shape[:2]

# ---------- 修改分區比例以符合機車視角 ----------
# 機車視角建議：左右區域較窄（例如 20%），前方區域擴大（y < 60% 為前方），
# 地面（近距離）定義為 y >= 60%（更靠近畫面下方代表接近車頭）
left_line_x = int(img_width * 0.20)   # 左側 20%
right_line_x = int(img_width * 0.80)  # 右側 80%
top_line_y = int(img_height * 0.60)   # 水平線往上（擴大前方範圍）
bottom_line_y = top_line_y

output_video_filename = f"./output/output_video_{date_time_string}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_filename, fourcc, 10.0, (img_width, img_height))

# detection loop variables
point_list = deque()
detection_info = []
bbox_list = []
key_frame = []
last_frame = 0
skipped_frame = 6
motion_factor = 10

object_list = []
object_alert = []

response_list = []
tokens_list = []
time_list = []

frame_list = []
frame_id = 0
start_time = time.time()

while True:
    frame_id += 1
    print('Current frame: ', frame_id)
    ret, frame = video_capture.read()
    if not ret:
        break
    if frame_id < display_start_frame:
        continue
    if frame_id > display_until_frame:
        break

    cv2.putText(frame,f'{frame_id-i}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[mark_danger], 2, cv2.LINE_AA)
    if frame_id % 5 in [1,2,3,4]:
        continue

    # yolo inference
    results = model.predict(frame)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    xywh = results[0].boxes.xywh
    mask = results[0].masks
    h,w = frame.shape[0:2]

    predictions = boxes.data.cpu().numpy()

    if len(predictions) > 0:
        if len(predictions[0]) == 7:
            boxes = predictions[:,0:4]
            tracker_id = predictions[:,4].astype(int) 
            classes = predictions[:,6].astype(int)
            scores = predictions[:,5]
        else:
            boxes = predictions[:,0:4]
            tracker_id = np.zeros(len(predictions)).astype(int)
            classes = predictions[:,5].astype(int)
            scores = predictions[:,4]
    else:
        continue

    scene=frame.copy()
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=classes,
        tracker_id = tracker_id
    )
    detections = detections[detections.confidence > 0.6]

    labels = [f'{tracker_id} {model.names[class_id]} {confidence:0.2f}'
            for confidence, class_id, tracker_id in zip(detections.confidence, 
                                                        detections.class_id, 
                                                        detections.tracker_id)]

    current_frame = [tracker_id, boxes, classes, scores]

    if frame_id > display_start_frame and frame_id % skipped_frame == 0 and skipped_frame>1:
        frame_info = []
        categorized_detections = {'frame_id':frame_id, 'left': [], 'right': [], 'front': [], 'ground': []}

        for pid, box, label, score in zip(tracker_id, boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(label)]
            if class_name not in object_list:
                object_list.append(class_name)

            height = y2-y1
            width = x2-x1
            center_x = x1 + (width)//2
            center_y = y1 + (height)//2

            # 將高度/寬度/中心轉成百分比（相對於影像尺寸）
            height_pct = int(height / h * 100)
            width_pct = int(width / w * 100)
            x_loc = int(center_x / w * 100)
            y_loc = int(center_y / h * 100)

            # size：以面積百分比量化（height_pct * width_pct / 100 ），代表相對面積程度
            size = int((height_pct * width_pct) / 100)

            # ---------- 分區判斷（機車視角） ----------
            # 左右區域收窄為 20%（更接近邊緣判定為側邊）
            if x_loc < 20:
                location = 'left'
            elif x_loc > 80:
                location = 'right'
            # 前方區域擴大到 y_loc < 60（機車視角前方重要）
            elif y_loc < 60:
                location = 'front'
            else:
                # 近距離地面/車頭附近判定
                location = 'ground'

            info = f'ID:{pid}, class:{class_name}, confidence:{score:.2f}, center_x:{x_loc}%, center_y:{y_loc}%, object_height:{height_pct}%, object_width:{width_pct}%, size: {size}%'
            categorized_detections[location].append(info)
            frame_info.append(info)

            # ---------- 警示規則（機車提高靈敏度） ----------
            # ground（近距離）全部加警示；側邊(left/right)只要相對面積超過 10% 就警示（原為 20%）
            if location == 'ground':
                object_alert.append([frame_id, location, pid, class_name, score, size])
            if location in ['left', 'right']:
                if size > 10:  # 閾值下修，提高側方警示靈敏度（機車速度較快）
                    object_alert.append([frame_id, location, pid, class_name, score, size])
            # 對 front 區域可以根據 size 或 confidence 設額外條件（這裡暫時不強制）
            if location == 'front' and size > 30 and score > 0.7:
                # 大型且接近的前方物體也加入警示
                object_alert.append([frame_id, location, pid, class_name, score, size])

        detection_info.append(categorized_detections)

        # 非同步啟動 GPT 評估（保留原邏輯）
        gpt_response_thread = threading.Thread(target=GPT_annotation, args=(categorized_detections,))
        gpt_response_thread.start()

        if len(GPT_list) > 0:
            [response, GPT_time_cost, usage] = GPT_list[-1]
            response_list.append(response)
            time_list.append(GPT_time_cost)

        try:
            GPT_data = eval(response)
            level = GPT_data['danger_score']
            content = GPT_data['reason']
        except: pass

    last_frame = [tracker_id, boxes, classes, scores]

    # 繪製分區線（以醒目粗線顯示）
    cv2.line(annotated_frame, (left_line_x, 0), (left_line_x, img_height), (0, 255, 0), 6)  # 左垂直線
    cv2.line(annotated_frame, (right_line_x, 0), (right_line_x, img_height), (0, 255, 0), 6)  # 右垂直線
    cv2.line(annotated_frame, (left_line_x, top_line_y), (right_line_x, bottom_line_y), (0, 255, 0), 6)  # 水平線

    if 'level' and 'content' in locals():
        text_1 = f"Emergency level: {level}"
        text_2 = content
        # color 根據 level 混合（0→綠, 1→紅）
        color = (0, int(255*(1-level)), int(255*level))
        text_x = left_line_x + 10
        text_y = img_height - 100
        cv2.putText(annotated_frame, text_1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(annotated_frame, text_2, (text_x, text_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Video Frame', annotated_frame)
    output_video.write(annotated_frame)

    time.sleep(1/fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('last.jpg', annotated_frame)
        break

video_capture.release()
output_video.release()
end_time = time.time()
total_time = end_time-start_time
FPS = (frame_id-display_start_frame)/total_time
print('FPS: ', FPS)

print('Unique Objects: ', len(object_list))
print('Danger Labels: ', len(object_alert))
