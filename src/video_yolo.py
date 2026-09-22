import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")
cap = cv2.VideoCapture("data/test1.mp4")

n = 0          # 第几帧
saved = 0      # 存了几张图

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 每 10 帧处理 1 帧，跳过其余
    if n % 10 != 0:
        n = n + 1
        continue

    results = model(frame, verbose=False)     # 直接喂这一帧的数组
    r = results[0]

    for b in r.boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])

        name = model.names[int(b.cls)]
        conf = float(b.conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1-8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(f"output/vedio_yolo/video_{saved}.jpg", frame)
    saved = saved + 1

    print(f"第 {n} 帧完成，已存 {saved} 张")

    n = n + 1

cap.release()
print("全部完成")
