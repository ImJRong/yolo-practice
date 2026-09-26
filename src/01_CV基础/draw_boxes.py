from ultralytics import YOLO
import cv2

img = cv2.imread("data/images/bus.png")

model = YOLO("models/yolov8m.pt")

results = model("data/images/bus.png", classes=[5])

result = results[0]
# 因为 box.xyxy 不是 [x1,y1,x2,y2]，它是装着 [x1,y1,x2,y2] 的盒子。
# box.xyxy            # tensor([[12.3, 45.6, 200.1, 300.7]])
#                              ↑ 注意两层方括号

for b in result.boxes:
    x1 = int(b.xyxy[0][0])
    y1 = int(b.xyxy[0][1])
    x2 = int(b.xyxy[0][2])
    y2 = int(b.xyxy[0][3])

    # x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]  ？？看不懂

    # img[y1:y1+5,x1:x2] = [0,0,255]
    # img[y2-5:y2,x1:x2] = [0,0,255]
    # img[y1:y2,x1:x1+5] = [0,0,255]
    # img[y1:y2,x2-5:x2] = [0,0,255]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)  # 画矩形（左上 右下 颜色 宽度）

    conf = float(b.conf)
    text = f"bus {conf:.2f}"
    cv2.putText(img, text, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2) #文本标记
    img_cv = cv2.imwrite("output/bus_cv.png", img)  #画框
    
