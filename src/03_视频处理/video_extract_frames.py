# ============================================
# 视频抽帧检测，每 10 帧存 1 张带框的图
#
# 思路：视频是一串图片，没必要每帧都处理
#   每 10 帧拿 1 帧 → 检测 → 画框 → 存成 jpg
#
# 跑法：在项目根目录执行  python src/video_yolo.py
# 输出：output/video_yolo/video_0.jpg ~ video_30.jpg
# ============================================

import cv2
from ultralytics import YOLO

import os

# 输出目录不存在的话，imwrite 会静默失败，先确保目录存在
os.makedirs("output/video_yolo", exist_ok=True)

model = YOLO("models/yolov8m.pt")

cap = cv2.VideoCapture("data/videos/test1.mp4")

if not cap.isOpened():
    print("打不开视频，检查路径")
    exit()

frame_id = 0   # 当前是第几帧
saved = 0      # 已经存了几张图

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 每 10 帧处理 1 帧，跳过其余
    # % 是取余数，frame_id % 10 != 0 就是"不是 10 的整数倍"
    # continue 跳过本轮剩下的代码，直接下一轮
    # 注意：跳过之前必须先 frame_id + 1，否则它永远不会变
    if frame_id % 10 != 0:
        frame_id = frame_id + 1
        continue

    results = model(frame, verbose=False)
    r = results[0]

    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()

        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])

        name = model.names[int(box.cls)]
        conf = float(box.conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text_y = max(y1 - 8, 20)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 用 saved 拼文件名，避免互相覆盖
    cv2.imwrite(f"output/video_yolo/video_{saved}.jpg", frame)
    saved = saved + 1

    print(f"第 {frame_id} 帧完成，已存 {saved} 张")

    frame_id = frame_id + 1

cap.release()
print("全部完成")
