# ============================================
# 视频实时播放检测 —— 不存文件，边播边画框
#
# 特点：自动适应视频尺寸，横屏 / 竖屏都能完整显示
#
# 跑法：在项目根目录执行  python src/video_play.py
# 退出：按 q 键
# ============================================

from ultralytics import YOLO
import cv2

# ============ 配置区（换视频只改这里） ============
VIDEO = "data/girl3.mp4"        # 输入视频
MODEL = "models/yolov8n.pt"     # 模型
MAX_W = 1200                    # 窗口最大宽度，超了就缩
MAX_H = 800                     # 窗口最大高度，超了就缩
# =================================================

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("打不开视频，检查路径：", VIDEO)
    exit()

# 读原视频尺寸
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# ---------- 算窗口该显示多大 ----------
# 思路：分别算"按高度缩"和"按宽度缩"的比例，取小的那个。
# 取小的才能保证宽和高两个方向都不超上限。
scale_h = MAX_H / height
scale_w = MAX_W / width
scale   = min(scale_h, scale_w)

if scale >= 1:
    # 原视频本来就比上限小，保持原样
    show_w = width
    show_h = height
else:
    show_w = int(width * scale)
    show_h = int(height * scale)

print(f"视频原始尺寸：{width} x {height}")
print(f"播放窗口尺寸：{show_w} x {show_h}")
print("按 q 退出")

index = 0

while True:
    isread, frame = cap.read()

    if not isread:                             # 读完了，退出循环
        break

    results = model(frame, verbose=False)      # 检测用的是原尺寸原图，精度不受影响
    frame = results[0].plot()                  # 画好框，返回 numpy 图

    # 只在需要时才缩放，省掉多余的计算
    if show_w != width or show_h != height:
        frame = cv2.resize(frame, (show_w, show_h))

    cv2.imshow("video", frame)                 # 弹窗显示这一帧

    # waitKey(1)：让画面刷新出来，同时接收按键
    #   1      = 只等 1 毫秒（给 0 会一直卡住等按键）
    #   0xFF   = 屏蔽高位，跨平台保险写法
    #   ord("q") = 字符 q 的编码 113
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    index = index + 1
    print(f"\r处理中 {index} 帧", end="")

cap.release()                                  # 释放视频
cv2.destroyAllWindows()                        # 关掉所有弹窗

print(f"\nDONE，共 {index} 帧")
