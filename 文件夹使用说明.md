# 文件夹使用说明

## 一句话

**代码里凡是提到文件，前面都要加文件夹名。**

---

## 三个必须加前缀的地方

### 1. 模型 → 加 `models/`

```python
model = YOLO("models/yolov8m.pt")     ✅
model = YOLO("yolov8m.pt")            ❌
```

**已有的模型文件**（都在 `models/`）：

```
models/yolov8n.pt         小，快
models/yolov8s.pt         中
models/yolov8m.pt         大，准（常用这个）
models/yolov8n-pose.pt    姿态估计
```

### 2. 输入图片 → 加 `data/`

```python
img = cv2.imread("data/bus.png")             ✅
results = model("data/test1.jpg")            ✅
img = cv2.imread("bus.png")                  ❌
```

**已有的图片**（都在 `data/`）：

```
data/bus.png       公交车
data/test.jpg      自拍
data/test1.jpg     自拍
data/image1.png    合照
data/image2.png
data/image3.png
data/image4.png
```

### 3. 输出图片 → 加 `output/`

```python
cv2.imwrite("output/结果.png", img)                    ✅
r.save(filename="output/result1.jpg")                  ✅
cv2.imwrite("结果.png", img)                           ❌
```

**`output/` 是自己生成的**，跑完去这个文件夹看结果图。

---

## 模板：新脚本照抄

```python
from ultralytics import YOLO
import cv2

img = cv2.imread("data/图片名.png")              # 输入
model = YOLO("models/yolov8m.pt")                # 模型

results = model("data/图片名.png")               # 检测

for r in results:
    r.save(filename="output/结果.png")           # 输出

cv2.imwrite("output/结果2.png", img)             # 输出
```

---

## 跑代码的位置

```powershell
cd D:\yolo-practice          # 必须在根目录!!!!!!!!
python src/test.py
```

**别 `cd src`** —— 那样 `data/`、`models/` 就找不到了。

---

## 检查清单

写完代码扫一眼，三个问题：

1. `YOLO(...)` 里的路径有没有 `models/`？
2. 读图的路径有没有 `data/`？
3. 存图的路径有没有 `output/`？

**三个都是"是"，就不会报路径错误。**

---

