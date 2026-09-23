# src 目录说明

## ⚠️ 运行纪律（最重要的一条）

**所有脚本都必须在项目根目录 `D:\yolo-practice` 下运行。**

```bat
cd D:\yolo-practice
python src\02_图片检测\detect_image.py
```

**为什么**：脚本里写的是相对路径（`data/xxx`、`models/xxx`、`output/xxx`）。
这些路径是相对 **「你敲命令时所在的目录」** 解析的，**不是**相对脚本文件在哪。

所以如果 `cd` 进 src 里面再执行，就会找不到 `data/`、`models/`，报"读不到文件"。

> 一句话：**永远站在项目根目录，`python src\...\xxx.py`。**

---

## 目录结构

```
src/
├── 01_CV基础/            OpenCV 操作图片（图片=数字表格）
├── 02_图片检测/           YOLO 单图 / 多图检测
├── 03_视频处理/           抽帧、逐帧检测、实时播放
└── 04_数据集与训练/        标注格式转换、数据划分、训练
```

---

## 01_CV基础

| 脚本 | 干什么 |
|---|---|
| `image_basics.py` | 看图片的本质：`shape`（高宽通道）、`dtype`（0~255）、取某个像素、缩放、存图 |
| `draw_boxes.py` | 用 `cv2.rectangle` 手动画 YOLO 检测框（练 `img[行,列]` 索引） |
| `draw_boxes_car.py` | 同上，换成了检测汽车 |

---

## 02_图片检测

| 脚本 | 干什么 |
|---|---|
| `detect_image.py` | 单张图检测，打印类别名 / 置信度 / 坐标，并把结果图存到 `output/` |
| `count_persons.py` | 单张图只检测人（`classes=[0]`）并数出人数 |
| `count_persons_batch.py` | 多张图一起检测并统计人数（用 `enumerate` 防止结果文件互相覆盖） |
| `explain_line_by_line.py` | 逐行讲解版，代码里每行都写了注释，用来对照学习 |

---

## 03_视频处理

| 脚本 | 干什么 |
|---|---|
| `video_extract_frames.py` | 视频抽帧：每 10 帧存一张图到 `output/video_yolo/` |
| `video_detect.py` | 视频逐帧检测，用 `r.plot()` 自动画框，存成新视频 |
| `video_detect_manual.py` | 同上，但用 `cv2.rectangle` + `cv2.putText` 手动画框（练手用） |
| `video_play.py` | **实时播放**：不存文件，边播边画框，按 `q` 退出。自动适配横竖屏 |
| `video_play_plus.py` | 实时播放的加强版 |

---

## 04_数据集与训练

| 脚本 | 干什么 |
|---|---|
| `dataset_fix_lane.py` | 把桌面「车道图」文件夹里的图统一转成 `.jpg` 并重命名为 `lane_01`~`lane_28` |
| `labels_xml_to_yolo.py` | **标注格式转换**：PascalVOC 的 `.xml` → YOLO 的 `.txt`（像素坐标 → 归一化中心点+宽高） |
| `dataset_split.py` | 划分训练集 / 验证集，生成 `data/lane/train.txt` 和 `val.txt` |
| `train_lane.py` | **训练车道线模型**（迁移学习，从 `models/yolov8n.pt` 开始） |

### 训练相关文件（不在 src 里）

| 文件 | 作用 |
|---|---|
| `data/lane/classes.txt` | 类别清单，一行 `lane` |
| `data/lane/data.yaml` | 数据配置单：告诉模型去哪找图片、有哪几类 |
| `data/lane/train.txt` / `val.txt` | 训练集 / 验证集的图片名单 |
| `data/lane/images/` | 图片（28 张） |
| `data/lane/labels/` | 标注文件 `.txt`（`_xml_backup/` 里是原始 xml 备份） |
| `output/lane_train/` | 训练产物：`weights/best.pt`（模型）、`results.png`（曲线图） |

---

## 一条完整链路（从零到有自己的模型）

```
1. 收集图片        →  data/lane/images/
2. 标注           →  双击桌面「标注车道线.bat」，W 画框，D 下一张
3. 格式转换        →  python src\04_数据集与训练\labels_xml_to_yolo.py
4. 划分训练/验证    →  python src\04_数据集与训练\dataset_split.py
5. 训练           →  python src\04_数据集与训练\train_lane.py
6. 用新模型检测     →  model = YOLO("output/lane_train/weights/best.pt")
```
