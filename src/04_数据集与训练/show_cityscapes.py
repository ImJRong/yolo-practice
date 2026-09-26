"""
把检测框画到图上，存成图片看效果

用途：
  拿模型在验证集上跑一遍，看图就知道它漏了什么、错在哪。
  现在用的是没训练过的 yolov8n，所以会看到漏很多 —— 对应召回率很低。

注意：
  yolov8n 只认识 COCO 的 80 类，这里只画我们要的 5 类，
  其他类别（狗、椅子、飞机…）全部不画。

跑法：在项目根目录执行  python src/04_数据集与训练/show_cityscapes.py
输出：output/cityscapes_show/
"""

import os
import glob
import random

import cv2
from ultralytics import YOLO

# ============ 配置区 ============
DATA_DIR = "data/cityscapes"                    # 数据集目录
OUT_DIR = "output/cityscapes_show"              # 结果存哪
MODEL_PATH = "models/yolov8n.pt"                # 用官方预训练模型，不训练

CLASSES = ["car", "person", "truck", "bus", "traffic light"]

NUM = 10        # 画几张
IMGSZ = 640     # 检测时缩放到的尺寸
CONF = 0.25     # 低于这个分数的框不要
IOU = 0.8       # 去重：两个框重叠超过 0.8 就只留一个
SEED = 42       # 固定随机种子，保证每次挑的图一样
MAX_W = 1400    # 原图 2048 太宽，存之前缩一下
# ===============================

# 每类一个颜色（OpenCV 是 BGR 顺序）
COLORS = {
    "car":           (0, 0, 255),      # 红
    "person":        (0, 255, 0),      # 绿
    "truck":         (255, 0, 0),      # 蓝
    "bus":           (0, 255, 255),    # 黄
    "traffic light": (255, 0, 255),    # 紫
}

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "images", "val", "*.png")))
random.seed(SEED)
picked = random.sample(img_paths, min(NUM, len(img_paths)))

for i, img_path in enumerate(picked, 1):
    img = cv2.imread(img_path)
    if img is None:
        continue

    r = model(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]

    n = 0
    for b in r.boxes:
        name = model.names[int(b.cls)]
        if name not in CLASSES:        # 不要的类别不画
            continue

        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        color = COLORS[name]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, "%s %.2f" % (name, float(b.conf)),
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        n += 1

    # 太宽的话缩一下再存
    h, w = img.shape[:2]
    if w > MAX_W:
        img = cv2.resize(img, (MAX_W, int(h * MAX_W / w)))

    name_out = "%02d_%s.jpg" % (i, os.path.basename(img_path)[:24])
    cv2.imwrite(os.path.join(OUT_DIR, name_out), img)
    print("%s  %d 个框" % (name_out, n))

print("输出目录：%s" % OUT_DIR)
