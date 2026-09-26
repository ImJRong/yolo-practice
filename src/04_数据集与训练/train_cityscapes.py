"""
训练 Cityscapes 检测模型

数据：data/cityscapes/（2962 训练 + 50 验证，5 类）
类别：car / person / truck / bus / traffic light

核心概念：迁移学习
  不从零开始，而是拿一个已经在 8 万张图上学过通用特征的模型（yolov8n.pt），
  在它的基础上继续学 Cityscapes 这 5 类。数据量少也能训出效果。

跑法：在项目根目录执行  python src/04_数据集与训练/train_cityscapes.py

      ⚠️ 别加 `| tail` 这类管道，会缓冲住输出、看不到实时进度。
      想看跑到第几轮，随时另开一个窗口执行：
        wc -l output/cityscape_train_v2/results.csv     （行数 - 1 = 已完成的轮数）

输出：output/cityscape_train_v2/
        weights/best.pt   —— 训练过程中最好的一份
        weights/last.pt   —— 最后一轮的（中途断了能从这续）
        results.csv       —— 每轮一行，记录 loss 和精度
        results.png       —— 曲线图
"""

from ultralytics import YOLO

# ============ 配置区 ============
DATA = "data/cityscapes/data.yaml"    # 数据配置单
BASE_MODEL = "models/yolov8n.pt"      # 迁移学习的起点。数据少要用小模型，大模型训不饱
EPOCHS = 35                          # 35轮
IMGSZ = 960                           # 图片统一缩放到 960 再喂给模型（640 会丢掉很小的交通灯）
BATCH = 4                             # 一次处理几张图，内存不够就调小
DEVICE = "cpu"                        # 用 CPU 训练（你没显卡）
# ===============================

model = YOLO(BASE_MODEL)

model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,

    # Windows 上多进程读数据容易报错，设 0 表示不用多进程（慢一点但稳）
    workers=0,

    # 结果输出位置。必须写【绝对路径】——
    # 写相对路径的话，ultralytics 会把它拼到 runs/detect/ 下面，就不是 output/ 了
    project=r"D:\yolo-practice\output",
    name="cityscape_train_v2",
)

print()
print("=" * 60)
print("训练结束。结果在 output/cityscape_train_v2/ 目录：")
print("  weights/best.pt   —— 训练过程中最好的那份模型")
print("  weights/last.pt   —— 最后一轮的模型（中途断了能从这续）")
print("  results.png       —— loss 和精度曲线图")
print("=" * 60)
