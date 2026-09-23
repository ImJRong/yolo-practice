"""
训练车道线检测模型

核心概念：迁移学习
  不从零开始训练，而是拿一个「已经在 8 万张图上学过通用特征」的模型（yolov8n.pt），
  在它的基础上继续学「车道线」这一件事。
  好处：数据量少也能训出效果，速度快几十倍。
  类比：不教一个婴儿从认字开始，而是找个会读书的人，教他认「车道线」这个词。

跑法：在项目根目录执行  python src/train_lane.py
输出：output/lane_train/ 目录下（权重文件 + 训练曲线 + 结果图表）
"""

from ultralytics import YOLO

# 加载预训练模型。models/ 目录下已经下载好了
model = YOLO("models/yolov8m.pt")

# 开始训练
model.train(
    data="data/lane/data.yaml",   # 数据配置单
    epochs=50,                    # 把所有图看几遍（第一次想快点看到结果，可以先改成 10）
    imgsz=640,                    # 图片统一缩放到 640 像素再喂给模型
    batch=4,                      # 一次处理几张图。显存/内存不够就调小
    device="cpu",                 # 用 CPU 训练（你没显卡）

    # Windows 上多进程读数据容易报错，设 0 表示不用多进程（慢一点但稳）
    workers=0,

    # 结果输出位置。必须写【绝对路径】——
    # 写相对路径的话，ultralytics 会把它拼到 runs/detect/ 下面，就不是 output/ 了
    project=r"D:\yolo-practice\output",
    name="lane_train",
)

print()
print("=" * 60)
print("训练结束。结果在 output/lane_train/ 目录：")
print("  weights/best.pt   —— 训练过程中最好的那份模型")
print("  weights/last.pt   —— 最后一轮的模型")
print("  results.png       —— loss 和精度曲线图")
print("=" * 60)
