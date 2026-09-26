# ============================================
# 用 ONNX 模型实时播放检测（视频版）
#
# 和 video_play_lane.py 的区别：推理引擎不同
#   video_play_lane.py → PyTorch 引擎（model(frame)），慢
#   video_onnx.py      → ONNX Runtime 引擎，实测快 2 倍
#
# 跑法：在项目根目录执行  python src/05_部署/video_onnx.py
# 退出：按 q
# ============================================

import time
import numpy as np
import cv2
import onnxruntime as ort

# ============ 配置区 ============
#   MODEL = "output/lane_train/weights/best.onnx"   #v1
MODEL = "output/lane_train_v2/weights/best.onnx"    #v2
VIDEO = "data/videos/NewModelTestV2.mp4"
SIZE = 640        # 和导出时一致
CONF = 0.3        # 分数门槛
IOU = 0.6        # 去重门槛
MAX_W = 1200      # 窗口最大宽
MAX_H = 800       # 窗口最大高
# ===============================

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("打不开视频：", VIDEO)
    exit()

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale = min(MAX_H / fh, MAX_W / fw)
show_w, show_h = (fw, fh) if scale >= 1 else (int(fw * scale), int(fh * scale))

# ---------- 预处理：把一帧图整理成模型要的形状 ----------
def preprocess(img):
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
    img = img.transpose(2, 0, 1)                 # 通道挪到最前
    img = img.astype(np.float32) / 255.0         # 0~255 → 0~1
    return np.expand_dims(img, 0)                # 加一维「1 张图」

# ---------- 后处理：从 8400 个候选框里挑出真框 ----------
def postprocess(out, fw, fh):
    pred = out[0].transpose()                    # (5, 8400) → (8400, 5)
    pred = pred[pred[:, 4] > CONF]               # 只留分数够的
    if len(pred) == 0:
        return []

    xywh = np.empty((len(pred), 4), dtype=np.float32)
    xywh[:, 0] = pred[:, 0] - pred[:, 2] / 2     # 中心x → 左上x
    xywh[:, 1] = pred[:, 1] - pred[:, 3] / 2     # 中心y → 左上y
    xywh[:, 2] = pred[:, 2]                      # 宽
    xywh[:, 3] = pred[:, 3]                      # 高

    idx = cv2.dnn.NMSBoxes(xywh.tolist(), pred[:, 4].tolist(), CONF, IOU)
    if len(idx) == 0:
        return []

    boxes = xywh[np.array(idx).flatten().astype(int)]
    # 坐标从 640 空间换算回原视频尺寸
    boxes[:, 0] *= fw / SIZE
    boxes[:, 2] *= fw / SIZE
    boxes[:, 1] *= fh / SIZE
    boxes[:, 3] *= fh / SIZE
    return boxes

index = 0
t0 = time.perf_counter()

while True:
    isread, frame = cap.read()
    if not isread:
        break

    x = preprocess(frame)
    out = session.run(None, {input_name: x})[0]
    boxes = postprocess(out, fw, fh)

    # 画框
    for b in boxes:
        x1, y1, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # 实时 FPS 显示在左上角
    index += 1
    fps = index / (time.perf_counter() - t0)
    cv2.putText(frame, "%.1f FPS" % fps, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if show_w != fw or show_h != fh:
        frame = cv2.resize(frame, (show_w, show_h))

    cv2.imshow("onnx lane", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("DONE，共 %d 帧，平均 %.1f FPS" % (index, index / (time.perf_counter() - t0)))
