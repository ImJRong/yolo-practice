# ============================================
# 检测一张图片，把识别结果打印出来
# 跑法：在项目根目录执行  python src/test.py
# ============================================

from ultralytics import YOLO

# 加载模型（从 models 文件夹拿）
model = YOLO("models/yolov8m.pt")

# 检测 data 文件夹里的图片
# 返回 results 是个列表，装了每张图的结果
results = model("data/images/test1.jpg")

# 遍历每张图的结果（这里只有一张）
for r in results:
    # 让 YOLO 把画好框的图存到 output
    r.save(filename="output/result1.jpg")

    print("--- detected objects ---")

    # 遍历这张图里检测到的每个物体
    for box in r.boxes:
        # 类别编号 → 类别名（0 = person，5 = bus）
        # box.cls 是张量，要 int() 转成整数才能当字典的键
        name = model.names[int(box.cls)]

        # 置信度，0~1，越接近 1 越确定
        conf = float(box.conf)

        # 框的坐标 [x1, y1, x2, y2]，左上角 + 右下角
        # box.xyxy 有外面一层壳，[0] 剥掉它
        # round() 把浮点数变成整数，方便看
        xyxy = [round(float(v)) for v in box.xyxy[0]]

        print(f"{name}  conf={conf:.2f}  box={xyxy}")

print("done")
