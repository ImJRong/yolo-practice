import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")
cap = cv2.VideoCapture("data/test1.mp4")

# 拿到原视频的尺寸和帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 建"写视频"的对象
# 参数：输出文件名, 编码器, 帧率, 画面尺寸(宽, 高)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output/video_yolo/test1.mp4", fourcc, fps, (w, h))

n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    r = results[0]

    for b in r.boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        name = model.names[int(b.cls)]
        conf = float(b.conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)          # 把这一帧塞进视频

    n = n + 1
    print(f"\r处理中 {n} 帧", end="")

cap.release()
out.release()                 # 必须调，不然文件是坏的
print(f"\n完成，输出到 output/video_yolo/test1.mp4")
