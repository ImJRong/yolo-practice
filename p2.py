from ultralytics import YOLO

model = YOLO("yolov8m.pt")

reasults = model(["image1.png","image2.png","image3.png","image4.png"],classes = [0])

count = 0

for i,r in enumerate(reasults):
    r.save(filename =f"reasult_{i+1}.png" )
    count = len(r.boxes)
    print(f"第{i+1}张图有{count}人")

print("Done")




