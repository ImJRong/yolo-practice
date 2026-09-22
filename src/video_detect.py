from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8m.pt")       #导入模型
video = cv2.VideoCapture("data/girl.mp4")       #导视频

if not video.isOpened():
    print("打不开视频，检查路径")
    exit()

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output/girl_detect.mp4",fourcc,fps,(width,height))

index = 1

while True:
    isread,frame = video.read()

    if not isread:
        break

    results = model(frame,verbose = False)
    frame = results[0].plot()
    writer.write(frame)
    print(f"已处理 {index} 帧")
    index = index + 1


print(f"DONE,共{index}帧")
video.release()
writer.release()