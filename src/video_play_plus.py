from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")              # 导入模型
cap = cv2.VideoCapture("data/girl3.mp4")        # 打开视频

if not cap.isOpened():
    print("打不开视频，检查路径")
    exit()

index = 0

while True:
    isread, frame = cap.read()

    if not isread:                             # 读完了，退出循环
        break

    results = model(frame, verbose=False, classes = [0])      # 只检测人
    frame = results[0].plot()                  # 画好框，返回 numpy 图

    # -------------------- 新东西 --------------------
    cv2.imshow("video", frame)                 # 弹窗显示这一帧

    # waitKey(1)：等 1 毫秒，同时接收按键，返回按键的 ASCII 码
    #   1   = 只等 1 毫秒（给 0 会一直卡住等按键）
    #   0xFF = 屏蔽高位，跨平台保险写法
    #   ord("q") = 把字符 q 转成 ASCII 码 113   按q退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n用户退出")
        break
    # -----------------------------------------------

    index = index + 1
    print(f"\r处理中 {index} 帧", end="")

# 释放资源
cap.release()

# 关掉所有 imshow 弹出来的窗口（不调的话窗口会残留）
cv2.destroyAllWindows()

print(f"\nDONE，共处理 {index} 帧")
