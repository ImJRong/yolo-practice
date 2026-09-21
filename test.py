from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model("test1.jpg")

for r in results:
    r.save(filename="result1.jpg")
    print("--- detected objects ---")
    for box in r.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        xyxy = [round(float(v)) for v in box.xyxy[0]]
        print(f"{name}  conf={conf:.2f}  box={xyxy}")

print("done")
