# ============================================
# 数一数：这张图里有几个人？
# 跑法：在项目根目录执行  python src/p1.py
# ============================================

from ultralytics import YOLO

# 加载模型
model = YOLO("models/yolov8m.pt")

# 检测图片，返回所有图的结果（列表）
results = model("data/images/image1.png")

# 只看第一张图的结果
r = results[0]

# 把画好框的图存下来
r.save(filename="output/p1_result.png")

# 计数器，从 0 开始
count = 0

# 遍历这张图里的每一个检测框
for box in r.boxes:
    # 取类别编号：box.cls 是张量，int() 转成整数
    cls_id = int(box.cls)

    # 编号 → 名字，比如 0 → "person"
    name = model.names[cls_id]

    # 如果这个框是"人"，计数器加 1
    if name == "person":
        count = count + 1

print(f"图里有{count}人")
