"""
把桌面「车道图」文件夹里的图片统一转成 YOLO 能用的格式：
  1. 不管原来是 png / webp / jpg，统一存成 .jpg
  2. 重命名成 lane_01.jpg、lane_02.jpg ...
  3. 输出到 data/lane/images/

⚠️ 关键点：cv2.imread 读不了中文路径（内部用 C++ fopen，不走 Windows Unicode API）。
   解决办法：先用 numpy.fromfile 把文件读成字节流，再用 cv2.imdecode 解码。
   写的时候同理：cv2.imencode 编成字节流，再用 .tofile 写盘。
"""

import os
import numpy as np
import cv2

SRC = r"C:\Users\RONGG\Desktop\车道图"
DST = r"D:\yolo-practice\data\lane\images"

os.makedirs(DST, exist_ok=True)

exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
files = [f for f in sorted(os.listdir(SRC)) if f.lower().endswith(exts)]

n = 0
ok_count = 0
fail_list = []
report = []

for f in files:
    n += 1
    src_path = os.path.join(SRC, f)
    dst_name = "lane_%02d.jpg" % n
    dst_path = os.path.join(DST, dst_name)

    # ---- 读：绕开中文路径 ----
    raw = np.fromfile(src_path, dtype=np.uint8)      # 按字节读整个文件
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)        # 从字节流解码成图像

    if img is None:
        fail_list.append(f)
        report.append("%-12s  读取失败：%s" % (dst_name, f[:44]))
        continue

    h, w = img.shape[0], img.shape[1]

    # ---- 写：同样绕开中文路径（这里 DST 是英文路径，但写法保持一致更保险）----
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        fail_list.append(f)
        report.append("%-12s  编码失败：%s" % (dst_name, f[:44]))
        continue
    buf.tofile(dst_path)

    ok_count += 1
    report.append("%-12s  <=  %-44s  %4d x %-4d" % (dst_name, f[:44], w, h))

print("=" * 76)
for line in report:
    print(line)
print("=" * 76)
print("共扫描 %d 张，成功 %d 张，失败 %d 张" % (n, ok_count, len(fail_list)))
print("输出目录：", DST)
