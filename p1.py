from ultralytics import YOLO
model = YOLO("yolov8m.pt")
reasult = model("image1.png")
# reasult = model("image.png",classes=[0])
r = reasult[0]
r.save(filename="p1_reasult.png") 
count = 0
for box in r.boxes:
    cls_id = int(box.cls)
    name = model.names[cls_id]
    if name == "person":
        count = count+1

print(f"图里有{count}人")

