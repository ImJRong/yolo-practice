from ultralytics import YOLO
import cv2

line_w = 1
word_w = 1
word_s = 0.4
red = (0,0,255)
blue = (255,0,0)


img = cv2.imread("data/car.png")

model = YOLO("models/yolov8m.pt")

results = model("data/car.png", classes=[2])

result = results[0]

for b in result.boxes:
    x1 = int(b.xyxy[0][0])
    y1 = int(b.xyxy[0][1])
    x2 = int(b.xyxy[0][2])
    y2 = int(b.xyxy[0][3])


    cv2.rectangle(img, (x1, y1), (x2, y2), red, line_w)  # 画矩形（左上 右下 颜色 宽度）

    conf = float(b.conf)
    text = f"car {conf:.2f}"
    cv2.putText(img, text, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, word_s, red, word_w) #文本标记

img_cv = cv2.imwrite("output/car_cv+.png", img)  
    
