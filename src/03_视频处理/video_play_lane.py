# ============================================
# 用【自己训练的模型】实时播放检测
#
# 和 video_play.py 的区别：只有模型不同
#   video_play.py       → 官方 yolov8n.pt，认识 80 类通用物体（人、车、狗…）
#   video_play_lane.py  → 你自己的 best.pt，只认识 1 类：lane（车道线）
#
# 跑法：在项目根目录执行  python src/03_视频处理/video_play_lane.py
# 退出：按 q 键
# ============================================

from ultralytics import YOLO
import cv2

# ============ 配置区 ============
VIDEO = "data/test1.mp4"                          # 想换视频改这里
MODEL = "output/lane_train/weights/best.pt"       # 你自己训的模型
MAX_W = 1200                                      # 窗口最大宽度
MAX_H = 800                                       # 窗口最大高度
# ===============================

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("打不开视频，检查路径：", VIDEO)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 算窗口尺寸：按比例缩到 MAX_W x MAX_H 以内
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

    results = model(frame, verbose=False)
    frame = results[0].plot()

    if show_w != width or show_h != height:
        frame = cv2.resize(frame, (show_w, show_h))

    cv2.imshow("lane model", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    index = index + 1
    print("\r处理中 %d 帧" % index, end="")

cap.release()
cv2.destroyAllWindows()

print("\nDONE，共 %d 帧" % index)
