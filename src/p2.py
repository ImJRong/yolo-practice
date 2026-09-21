# ============================================
# 一次检测多张图片，分别统计人数、分别保存
# 跑法：在项目根目录执行  python src/p2.py
# ============================================

from ultralytics import YOLO

# 加载模型
model = YOLO("models/yolov8m.pt")

# 一次喂进 4 张图
# classes=[0] 表示只检测"人"，其他类别忽略
results = model([
    "data/image1.png",
    "data/image2.png",
    "data/image3.png",
    "data/image4.png",
], classes=[0])

# enumerate 会同时给出"序号 i"和"结果 r"
# i 从 0 开始：0, 1, 2, 3
for i, r in enumerate(results):
    # 用 i 拼文件名，避免多张图互相覆盖
    # 比如 output/result_1.png、output/result_2.png
    r.save(filename=f"output/result_{i+1}.png")

    # len() 数一下这张图有几个框（也就是几个人）
    count = len(r.boxes)

    print(f"第{i+1}张图有{count}人")

print("Done")
