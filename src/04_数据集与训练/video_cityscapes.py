"""
用【刚训好的 Cityscapes 模型】实时播放检测

和 video_play_lane.py 的区别：只有模型和类别不同
  video_play_lane.py     → 车道线模型，1 类
  video_cityscapes.py    → Cityscapes 模型，5 类（车/人/卡车/公交/交通灯）

跑法：在项目根目录执行  python src/04_数据集与训练/video_cityscapes.py
退出：按 q 键
"""

from ultralytics import YOLO
import cv2

# ============ 配置区 ============
VIDEO = "data/videos/NewModelTestV2.mp4"                 # 想换视频改这里
MODEL = "output/cityscape_train_v2/weights/best.pt"  # 刚训好的 Cityscapes 模型

CONF = 0.4        # 低于这个分数的框不要
IOU = 0.5         # 去重：重叠超过 0.5 只留一个
IMGSZ = 960       # 检测时缩放到多大（和训练时一致）

MAX_W = 1200      # 窗口最大宽度
MAX_H = 800       # 窗口最大高度
# ===============================

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("打不开视频，检查路径：", VIDEO)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 算窗口尺寸：按比例缩到 MAX_W x MAX_H 以内（只缩小，不放大）
scale = min(MAX_H / height, MAX_W / width)
if scale >= 1:
    show_w, show_h = width, height
else:
    show_w = int(width * scale)
    show_h = int(height * scale)

print("视频尺寸：%d x %d" % (width, height))
print("窗口尺寸：%d x %d" % (show_w, show_h))
print("按 q 退出")

index = 0

while True:
    isread, frame = cap.read()
    if not isread:
        break

    results = model(frame, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
    frame = results[0].plot()

    if show_w != width or show_h != height:
        frame = cv2.resize(frame, (show_w, show_h))

    cv2.imshow("cityscapes model", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    index = index + 1
    print("\r处理中 %d 帧" % index, end="")

cap.release()
cv2.destroyAllWindows()

print("\nDONE，共 %d 帧" % index)
