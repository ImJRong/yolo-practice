"""
算基线分：用【没训练过的】yolov8n 在 Cityscapes 验证集上跑一遍，看 P 和 R

为什么要算：
  后面用自己的模型再算一次，两个分一比，才知道训练到底有没有用。

为什么不用 model.val()：
  yolov8n 是在 COCO 上训的，认识 80 类，编号跟我们这 5 类对不上
  （COCO 里 car 是 2 号，我们这里是 0 号）。所以要按「名字」翻译一遍。
  model.val() 是黑盒，插不进这一步。

怎么算分：
  预测框和答案框 类别相同 且 重叠度 >= HIT_IOU，算命中
  精确率 P = 命中 / 报出     （别乱报）
  召回率 R = 命中 / 答案     （别漏）

跑法：在项目根目录执行  python src/04_数据集与训练/baseline_cityscapes.py
"""

import os
import glob

import cv2
from ultralytics import YOLO

# ============ 配置区 ============
DATA_DIR = "data/cityscapes"        # 数据集目录
MODEL_PATH = "models/yolov8n.pt"    # 用官方预训练模型，不训练

CLASSES = ["car", "person", "truck", "bus", "traffic light"]

IMGSZ = 640        # 检测时缩放到的尺寸
CONF = 0.25        # 低于这个分数的框不要
IOU = 0.8          # 去重：两个框重叠超过 0.8 就只留一个
HIT_IOU = 0.5      # 和答案重叠到这个程度算命中
# ===============================


def load_gt(txt_path, w, h):
    """读答案文件，把 0~1 的坐标还原成像素坐标"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            xc, yc, bw, bh = [float(v) for v in p[1:5]]
            boxes.append((cid,
                          (xc - bw / 2) * w, (yc - bh / 2) * h,
                          (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def iou(a, b):
    """两个框的重叠度。1 = 完全重合，0 = 不沾边"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


model = YOLO(MODEL_PATH)

# COCO 的类别名 → 我们的编号。COCO 里没有的名字直接丢掉
name2id = {name: CLASSES.index(name)
           for _, name in model.names.items() if name in CLASSES}

img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "images", "val", "*.png")))

# 每类一个 [命中, 报出, 答案]
stat = {c: [0, 0, 0] for c in CLASSES}

for img_path in img_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    gt = load_gt(os.path.join(DATA_DIR, "labels", "val", base + ".txt"), w, h)

    r = model(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]

    # 把预测的类别名翻译成我们的编号
    pred = []
    for b in r.boxes:
        name = model.names[int(b.cls)]
        if name in name2id:
            pred.append((name2id[name],) + tuple(float(v) for v in b.xyxy[0]))

    for cid, cname in enumerate(CLASSES):
        ps = [p for p in pred if p[0] == cid]
        gs = [g for g in gt if g[0] == cid]
        stat[cname][1] += len(ps)      # 报出
        stat[cname][2] += len(gs)      # 答案

        used = set()                   # 一个答案框只算一次命中
        for p in ps:
            best, bj = 0.0, -1
            for j, g in enumerate(gs):
                if j in used:
                    continue
                v = iou(p[1:], g[1:])
                if v > best:
                    best, bj = v, j
            if best >= HIT_IOU:
                used.add(bj)
                stat[cname][0] += 1

print("%-14s %6s %6s %6s %8s %8s" % ("类别", "答案", "报出", "命中", "精确率", "召回率"))

P, R = [], []
for cname in CLASSES:
    hit, npred, ngt = stat[cname]
    p = hit / npred if npred else 0.0
    r_ = hit / ngt if ngt else 0.0
    P.append(p)
    R.append(r_)
    print("%-14s %6d %6d %6d %8.3f %8.3f" % (cname, ngt, npred, hit, p, r_))

print("%-14s %6s %6s %6s %8.3f %8.3f" % ("平均", "", "", "", sum(P) / len(P), sum(R) / len(R)))
