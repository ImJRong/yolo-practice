# ============================================
# 视频实时播放检测 —— 不存文件，边播边画框
#
# 和 video_detect.py 的区别：
#   video_detect.py  → 每帧写进新视频文件（存起来慢慢看）
#   video_play.py    → 每帧直接弹窗显示（实时看）
#
# 跑法：在项目根目录执行  python src/video_play.py
# 退出：按 q 键
# ============================================

from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")              # 导入模型
cap = cv2.VideoCapture("data/girl.mp4")        # 打开视频

if not cap.isOpened():
    print("打不开视频，检查路径")
    exit()

index = 0

while True:
    isread, frame = cap.read()

    if not isread:                             # 读完了，退出循环
        break

    results = model(frame, verbose=False)      # 检测
    frame = results[0].plot()                  # 画好框，返回 numpy 图

    # -------------------- 新东西 --------------------
    cv2.imshow("video", frame)                 # 弹窗显示这一帧

    # waitKey(1)：等 1 毫秒，同时接收按键，返回按键的 ASCII 码
    #   1   = 只等 1 毫秒（给 0 会一直卡住等按键）
    #   0xFF = 屏蔽高位，跨平台保险写法
    #   ord("q") = 把字符 q 转成 ASCII 码 113
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # -----------------------------------------------

    index = index + 1
    print(f"\r处理中 {index} 帧", end="")

# 释放资源
cap.release()

# 关掉所有 imshow 弹出来的窗口（不调的话窗口会残留）
cv2.destroyAllWindows()

print(f"\nDONE，共 {index} 帧")
