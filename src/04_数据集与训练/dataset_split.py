"""
划分训练集 / 验证集，生成 train.txt 和 val.txt

为什么要划分：
  训练时模型只看训练集。要判断它「有没有真的学会」，必须用一批它没见过的图去考它，
  这批图就是验证集。如果训练和验证用同一批图，等于考前把答案给学生 —— 分数虚高，
  看不出真实水平。

划分规则：
  只取「已标注」的图片，按 4:1 分成训练集和验证集。
  固定随机种子（seed），保证每次跑结果一样，可复现。

跑法：在项目根目录执行  python src/dataset_split.py
输出：data/lane/train.txt、data/lane/val.txt（每行一个图片路径）
"""

import os
import random

LANE_DIR = r"D:\yolo-practice\data\lane"
IMAGES_DIR = os.path.join(LANE_DIR, "images")
LABELS_DIR = os.path.join(LANE_DIR, "labels")

VAL_RATIO = 0.2          # 验证集占比（20%）
SEED = 42                # 固定种子，保证划分可复现


def main():
    all_imgs = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg"))

    # 只要「图 + 标注文件」都存在的，没标注的图不参与训练
    labeled = []
    skipped = []
    for img in all_imgs:
        base = os.path.splitext(img)[0]
        if os.path.exists(os.path.join(LABELS_DIR, base + ".txt")):
            labeled.append(img)
        else:
            skipped.append(img)

    if len(labeled) < 2:
        print("已标注的图片不足 2 张，无法划分")
        return

    random.seed(SEED)
    random.shuffle(labeled)

    n_val = max(1, int(len(labeled) * VAL_RATIO))
    val_imgs = sorted(labeled[:n_val])
    train_imgs = sorted(labeled[n_val:])

    def write_list(name, imgs):
        path = os.path.join(LANE_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for img in imgs:
                # 写绝对路径，最不容易出错
                f.write(os.path.join(IMAGES_DIR, img) + "\n")
        return path

    p_train = write_list("train.txt", train_imgs)
    p_val = write_list("val.txt", val_imgs)

    print("=" * 60)
    print("已标注图片：%d 张" % len(labeled))
    print("  训练集：%d 张  ->  %s" % (len(train_imgs), p_train))
    print("  验证集：%d 张  ->  %s" % (len(val_imgs), p_val))
    if skipped:
        print()
        print("跳过（没有标注文件）：%d 张" % len(skipped))
        for s in skipped:
            print("  " + s)
    print("=" * 60)
    print()
    print("验证集内容：")
    for v in val_imgs:
        print("  " + v)


if __name__ == "__main__":
    main()
