"""
把 LabelImg 存成的 PascalVOC 格式（.xml）转成 YOLO 需要的格式（.txt）

为什么要转：
  LabelImg 默认存 PascalVOC 格式，坐标是「像素绝对值」（如 xmin=796）。
  YOLO 要的是「归一化后的中心点 + 宽高」，取值范围 0~1。

两种格式对比
  假设图片宽 1932、高 1280，框住 xmin=796 ymin=468 xmax=1932 ymax=1097：

  PascalVOC： <xmin>796</xmin> <ymin>468</ymin> <xmax>1932</xmax> <ymax>1097</ymax>
  YOLO：     0 0.706522 0.611328 0.587992 0.491406
             ↑  ↑        ↑        ↑        ↑
          类别  中心x     中心y     宽       高   （全部是 0~1 的小数）

换算公式
  x_center = (xmin + xmax) / 2 / 图片宽
  y_center = (ymin + ymax) / 2 / 图片高
  width    = (xmax - xmin) / 图片宽
  height   = (ymax - ymin) / 图片高

跑法
  在项目根目录执行：python src/labels_xml_to_yolo.py
输出
  labels/ 下生成同名 .txt 文件；原来的 .xml 移到 labels/_xml_backup/ 备份（不删）
"""

import os
import glob
import shutil
import xml.etree.ElementTree as ET

LABELS_DIR = r"D:\yolo-practice\data\lane\labels"
BACKUP_DIR = os.path.join(LABELS_DIR, "_xml_backup")

# 类别清单。顺序决定 class_id：第一个是 0，第二个是 1……
# 必须和 data/lane/classes.txt 里的顺序完全一致
CLASSES = ["lane"]


def convert_one(xml_path):
    """
    把一个 xml 转成 YOLO 格式的行（字符串列表）。
    返回 (行列表, 警告列表)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读图片尺寸。归一化要用它当分母
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    warns = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()

        # 类别不在清单里就跳过，否则 class_id 会跟训练时对不上
        if name not in CLASSES:
            warns.append("未知类别 %r，已跳过" % name)
            continue
        class_id = CLASSES.index(name)

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # 裁剪到图片范围内。越界会让归一化值 >1，训练时直接报错
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            warns.append("这个框宽或高为 0，已跳过")
            continue

        # 归一化
        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        # 再兜一层检查，确保落在 0~1
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            warns.append("归一化后越界，已跳过")
            continue

        lines.append("%d %.6f %.6f %.6f %.6f" % (class_id, xc, yc, bw, bh))

    return lines, warns


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.xml")))
    if not xml_files:
        print("没找到 .xml 文件，无需转换")
        return

    print("找到 %d 个 xml 文件" % len(xml_files))
    print("=" * 72)

    total_box = 0
    all_warns = []

    for xml_path in xml_files:
        base = os.path.splitext(os.path.basename(xml_path))[0]
        lines, warns = convert_one(xml_path)

        txt_path = os.path.join(LABELS_DIR, base + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        # 原 xml 移到备份目录，避免和 txt 混在一起
        shutil.move(xml_path, os.path.join(BACKUP_DIR, os.path.basename(xml_path)))

        total_box += len(lines)
        print("%-14s  %2d 个框  ->  %s.txt" % (base, len(lines), base))
        for w in warns:
            all_warns.append("%s: %s" % (base, w))

    print("=" * 72)
    print("共处理 %d 个文件，%d 个标注框" % (len(xml_files), total_box))
    print("原 xml 已备份到：%s" % BACKUP_DIR)

    if all_warns:
        print()
        print("警告：")
        for w in all_warns:
            print("  " + w)


if __name__ == "__main__":
    main()
