# ============================================
# 用 ONNX Runtime 跑自己的模型，三段分开计时
#
# 和 video_play_lane.py 的区别：
#   那个用 model(frame) → PyTorch 引擎，慢、包大
#   这个用 onnxruntime   → 专用推理引擎，实测快 2.1 倍
#
# 跑法：在项目根目录执行  python src/05_部署/benchmark_onnx.py
# ============================================

import time
import numpy as np
import cv2
import onnxruntime as ort

# ============ 配置区 ============
# MODEL = "output/lane_train/weights/best.onnx" #v1
MODEL = "output/lane_train_v2/weights/best.onnx"   #v2
IMAGE = "data/lane/images/lane_01.jpg"
SIZE = 640        # 导出时定的输入尺寸，必须一致
CONF = 0.5        # 分数门槛
IOU = 0.45        # 去重门槛
RUNS = 30         # 测多少次取平均
# ===============================

# ---------- 第 1 步：打开 ONNX 模型 ----------
session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("输入：", session.get_inputs()[0].shape)    # [1, 3, 640, 640]
print("输出：", session.get_outputs()[0].shape)   # [1, 5, 8400]

# ---------- 第 2 步：读图（防中文路径，老办法） ----------
img = cv2.imdecode(np.fromfile(IMAGE, dtype=np.uint8), cv2.IMREAD_COLOR)
print("原图：", img.shape)

# ---------- 第 3 步：预处理 ----------
def preprocess(img):
    img = cv2.resize(img, (SIZE, SIZE))          # 缩到 640x640
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # OpenCV 默认 BGR，模型要 RGB
    img = img.transpose(2, 0, 1)                 # 「高宽通道」→「通道高宽」
    img = img.astype(np.float32) / 255.0         # 0~255 压到 0~1
    img = np.expand_dims(img, 0)                 # 前面加一维，表示「1 张图」
    return img

# ---------- 第 4 步：后处理 ----------
def postprocess(out):
    pred = out[0].transpose()                    # (5, 8400) → (8400, 5)
    pred = pred[pred[:, 4] > CONF]               # 只留分数够的
    if len(pred) == 0:
        return []

    xywh = np.empty((len(pred), 4), dtype=np.float32)
    xywh[:, 0] = pred[:, 0] - pred[:, 2] / 2     # 中心x → 左上角x
    xywh[:, 1] = pred[:, 1] - pred[:, 3] / 2     # 中心y → 左上角y
    xywh[:, 2] = pred[:, 2]                      # 宽
    xywh[:, 3] = pred[:, 3]                      # 高

    idx = cv2.dnn.NMSBoxes(xywh.tolist(), pred[:, 4].tolist(), CONF, IOU)
    if len(idx) == 0:
        return []
    return xywh[np.array(idx).flatten().astype(int)]

# ---------- 第 5 步：预热 ----------
# 第一次跑总是慢（要分配内存、加载算子），先空跑几次把结果丢掉
x = preprocess(img)
for _ in range(3):
    session.run(None, {input_name: x})

# ---------- 第 6 步：正式计时 ----------
t_pre = t_inf = t_post = 0.0
for _ in range(RUNS):
    t0 = time.perf_counter()
    x = preprocess(img)
    t1 = time.perf_counter()
    out = session.run(None, {input_name: x})[0]
    t2 = time.perf_counter()
    boxes = postprocess(out)
    t3 = time.perf_counter()
    t_pre  += t1 - t0
    t_inf  += t2 - t1
    t_post += t3 - t2

t_pre  = t_pre  / RUNS * 1000
t_inf  = t_inf  / RUNS * 1000
t_post = t_post / RUNS * 1000
total  = t_pre + t_inf + t_post

print()
print("检出框数 ：%d 个" % len(boxes))
print("预处理  ：%6.2f ms" % t_pre)
print("推理    ：%6.2f ms" % t_inf)
print("后处理  ：%6.2f ms" % t_post)
print("-" * 28)
print("单帧合计：%6.2f ms  =>  %.1f FPS" % (total, 1000 / total))
