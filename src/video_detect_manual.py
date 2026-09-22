# ============================================
# 视频逐帧检测，生成带检测框的新视频
#
# 思路：视频 = 一串图片（305 帧 = 305 张图）
#   循环：读一帧 → 检测 → 画框 → 写进新视频，直到读完
#
# 跑法：在项目根目录执行  python src/video_detect.py
# 输出：output/video_yolo/test1.mp4
# ============================================

import cv2
from ultralytics import YOLO

import os

# 输出目录不存在时 VideoWriter 会静默失败（不报错，但视频是空的），先确保目录在
os.makedirs("output/video_yolo", exist_ok=True)

model = YOLO("models/yolov8m.pt")

# 打开视频。cap 是操作视频的对象，之后都通过它来
cap = cv2.VideoCapture("data/test1.mp4")

# isOpened() 检查是否打开成功（路径错、文件不在会返回 False）
if not cap.isOpened():
    print("打不开视频，检查路径")
    exit()

# cap.get(常量) 读取视频参数。宽高一定是整数（像素没有半个）
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 宽
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
fps = cap.get(cv2.CAP_PROP_FPS)                   # 帧率（每秒几帧）

# fourcc 指定压缩方法。"mp4v" 是一种视频压缩格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 创建写视频的对象（空视频容器）。四个参数：文件名、压缩格式、帧率、尺寸
# 注意尺寸是 (宽, 高)，与 img.shape 的 (高, 宽) 相反
writer = cv2.VideoWriter("output/video_yolo/test1.mp4", fourcc, fps, (width, height))

frame_id = 0   # 已处理的帧数

while True:
    # read() 读一帧，返回两个值：
    #   ret   = 是否读到（True / False，读完为 False）
    #   frame = 这一帧的图像数据
    ret, frame = cap.read()

    if not ret:      # 读完了，退出循环
        break

    results = model(frame, verbose=False)       #verbose=False：不打印检测进度
    r = results[0]

    # 画框
    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()

        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])

        name = model.names[int(box.cls)]
        conf = float(box.conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 文字上移 8 像素，max() 防止超出画面上方（最小留 20）
        text_y = max(y1 - 8, 20)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # 把这一帧写进新视频
    writer.write(frame)

    # \r 回到行首，end="" 不换行，让进度在一行内刷新
    frame_id = frame_id + 1
    print(f"\r处理中 {frame_id} 帧", end="")

# 释放资源。writer.release() 必须调，否则视频文件不完整
cap.release()
writer.release()

print(f"\n完成，共 {frame_id} 帧，输出到 output/video_yolo/test1.mp4")
