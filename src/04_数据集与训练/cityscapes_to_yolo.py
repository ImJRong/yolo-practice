"""
把 Cityscapes 的标注（多边形）转成 YOLO 要的格式（归一化矩形框）

为什么要转：
  Cityscapes 的标注记的是多边形（一圈顶点，用来描物体轮廓），
  YOLO 要的是矩形框 + 归一化坐标。所以要取多边形的「外接矩形」再换算。

换算公式（和 labels_xml_to_yolo.py 是同一套，只是输入不同）：
  x_center = (xmin + xmax) / 2 / 图片宽
  y_center = (ymin + ymax) / 2 / 图片高
  width    = (xmax - xmin) / 图片宽
  height   = (ymax - ymin) / 图片高

类别处理：
  Cityscapes 有 30 种标签，大部分是「区域面」（road、sky、building…），
  不是一个个人物或车，做成框没意义，全部跳过。只保留下面 CLASSES 这几类。

跑法：在项目根目录执行  python src/04_数据集与训练/cityscapes_to_yolo.py
输出：data/cityscapes/
        images/train/   labels/train/
        images/val/     labels/val/
        data.yaml
"""

import os
import json
import glob
import shutil

# ============ 配置区 ============
RAW_DIR = "data/cityscapes_raw"     # 下载的原始数据放这
OUT_DIR = "data/cityscapes"         # 转好的数据集放这

# 只保留这些类别。顺序就是 class_id：第 1 个是 0，第 2 个是 1……
# 选这 5 类的原因：COCO 预训练模型（yolov8n.pt）里正好有同名的，
# 名字能对上，才能拿它算基线分做对比。
CLASSES = [
    "car",
    "person",
    "truck",
    "bus",
    "traffic light",
]

TRAIN_LIMIT = None   # train 取全部（2975 张）。想只用前 N 张就改成数字
VAL_LIMIT = 50       # val 只留 50 张，够算分就行
# ===============================


def read_json(json_path):
    """读一个标注 json，返回 (图片宽, 图片高, 物体列表)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["imgWidth"], data["imgHeight"], data["objects"]


def to_yolo_line(obj, img_w, img_h):
    """一个多边形 → 一行 YOLO 标注。不在清单里或框不合法就返回 None"""
    name = obj["label"]
    if name not in CLASSES:
        return None, name

    # 多边形 → 外接矩形：取所有顶点里 x 最小/最大、y 最小/最大
    xs = [p[0] for p in obj["polygon"]]
    ys = [p[1] for p in obj["polygon"]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 裁到图片范围内。越界会让归一化值 >1，训练时直接报错
    xmin = max(0.0, min(float(xmin), img_w))
    xmax = max(0.0, min(float(xmax), img_w))
    ymin = max(0.0, min(float(ymin), img_h))
    ymax = max(0.0, min(float(ymax), img_h))

    if xmax <= xmin or ymax <= ymin:
        return None, name

    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h

    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        return None, name

    return "%d %.6f %.6f %.6f %.6f" % (CLASSES.index(name), xc, yc, bw, bh), name


def process_split(split, limit):
    """处理一个 split（train 或 val）"""
    img_dir = os.path.join(OUT_DIR, "images", split)
    lbl_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 标注文件按城市分子文件夹放
    pattern = os.path.join(RAW_DIR, "gtFine", split, "*", "*_gtFine_polygons.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print("  [%s] 没找到标注文件，跳过" % split)
        return 0, 0, {}

    if limit is not None:
        json_files = json_files[:limit]

    total_box = 0
    used = 0
    skip_stat = {}

    for json_path in json_files:
        # 从标注文件名推出图片文件名：
        #   aachen_000000_000019_gtFine_polygons.json
        #   → aachen_000000_000019_leftImg8bit.png
        base = os.path.basename(json_path).replace("_gtFine_polygons.json", "")
        city = os.path.basename(os.path.dirname(json_path))
        img_name = base + "_leftImg8bit.png"

        img_src = os.path.join(RAW_DIR, "leftImg8bit", split, city, img_name)
        if not os.path.exists(img_src):
            print("  [警告] 找不到照片：%s" % img_src)
            continue

        img_w, img_h, objects = read_json(json_path)

        lines = []
        for obj in objects:
            line, name = to_yolo_line(obj, img_w, img_h)
            if line is None:
                skip_stat[name] = skip_stat.get(name, 0) + 1
            else:
                lines.append(line)

        # 一张图里一个要的类别都没有，就整张跳过
        if not lines:
            continue

        shutil.copy(img_src, os.path.join(img_dir, img_name))

        txt_name = base + "_leftImg8bit.txt"
        with open(os.path.join(lbl_dir, txt_name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        total_box += len(lines)
        used += 1

    return used, total_box, skip_stat


def clear_cache():
    """
    删掉旧的标注缓存。

    为什么只删缓存、不清空整个目录：
      同名文件会被直接覆盖，所以重复跑不会出错（这次从 300 张放成全量，前 300 张原样重写）。
      注意：反过来把 TRAIN_LIMIT 调小时，多出来的旧图不会自动消失，得手动删。
    """
    for cache in ["labels/train.cache", "labels/val.cache"]:
        p = os.path.join(OUT_DIR, cache)
        if os.path.exists(p):
            os.remove(p)


def write_yaml():
    """生成 data.yaml"""
    path = os.path.join(OUT_DIR, "data.yaml")
    abs_dir = os.path.abspath(OUT_DIR).replace("\\", "/")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cityscapes 转出来的检测数据集\n")
        f.write("# 类别编号必须和 labels 里每行的第一个数字对应\n\n")
        f.write("path: %s\n" % abs_dir)
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    return path


# ---------- 主流程 ----------
print("=" * 72)
print("类别（共 %d 个）：" % len(CLASSES))
for i, c in enumerate(CLASSES):
    print("  %d = %s" % (i, c))
print("=" * 72)

if TRAIN_LIMIT is not None:
    print("提示：train 只取前 %d 张（TRAIN_LIMIT）" % TRAIN_LIMIT)
else:
    print("提示：train 取全部")

# 先清掉旧的标注缓存，否则 YOLO 会拿上次的列表算分
print("清理旧缓存…")
clear_cache()

all_skip = {}

for split, limit in [("train", TRAIN_LIMIT), ("val", VAL_LIMIT)]:
    print()
    print("--- 处理 %s ---" % split)
    used, boxes, skip_stat = process_split(split, limit)
    print("  可用图片：%d 张，标注框：%d 个" % (used, boxes))
    for k, v in skip_stat.items():
        all_skip[k] = all_skip.get(k, 0) + v

yaml_path = write_yaml()

print()
print("=" * 72)
print("完成。输出目录：%s" % OUT_DIR)
print("  images/train/  labels/train/")
print("  images/val/    labels/val/")
print("  data.yaml  ->  %s" % yaml_path)
print()
print("跳过的类别（不是一个个物体，或不在清单里）：")
for k in sorted(all_skip):
    print("  %-16s %d 次" % (k, all_skip[k]))
print("=" * 72)
