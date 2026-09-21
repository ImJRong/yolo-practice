# yolo-practice

YOLO + OpenCV 学习项目（武汉理工大学 · 电子信息 研一）

## 目录结构

```
yolo-practice/
├── models/       模型权重（*.pt，不上传 git）
├── data/         测试图片
├── src/          所有 Python 脚本
├── output/       检测结果图（不上传 git）
├── .vscode/      VS Code 配置
├── .gitignore
├── start-yolo.bat   一键进环境
└── 学习日志.md      学习记录
```

## 怎么跑

**所有命令都从项目根目录执行**（`D:\yolo-practice`）：

```powershell
python src/test.py
python src/cv_basics.py
python src/cv_draw.py
```

不要 `cd src` 再跑，否则里面的相对路径会失效。

也可以双击桌面的 `YOLO 终端.lnk`，自动进环境并切到项目根目录。

## 环境

- conda 环境：`yolo`（Python 3.11）
- 提示符要出现 `(yolo)` 才算激活成功
- 解释器：`D:\miniconda3\envs\yolo\python.exe`

## 路径约定

脚本里的路径一律相对**项目根目录**写：

```python
model = YOLO("models/yolov8m.pt")        # 模型
img = cv2.imread("data/bus.png")         # 输入图
cv2.imwrite("output/result.png", img)    # 输出图
```

## 脚本说明

| 文件 | 干什么 |
|---|---|
| `src/test.py` | 检测单张图，打印类别/置信度/坐标 |
| `src/p1.py` | 数一张图里有几个人 |
| `src/p2.py` | 批量检测 4 张图，分别保存结果 |
| `src/cv_basics.py` | OpenCV 基础：形状、索引、切片、缩放、灰度 |
| `src/cv_draw.py` | YOLO 检测 + OpenCV 手动画框 |
| `src/day2_explain.py` | 逐行讲解版（学习用） |

## Git

- 远程：`git@github.com:ImJRong/yolo-practice.git`（Private）
- `*.pt` 和 `output/` 已被 `.gitignore` 排除
