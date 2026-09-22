# YOLO 学习项目（D:\yolo-practice）长期笔记

## 环境
- 项目根：`D:\yolo-practice`（GitHub Private：`git@github.com:ImJRong/yolo-practice.git`）
- Conda：`D:\miniconda3`，环境名 `yolo`（python 3.11）
- 实测版本：opencv-python 5.0.0 / torch 2.14.0+cpu / ultralytics 8.4.157 / numpy 2.4.6
- 跑代码：`D:/miniconda3/Scripts/activate` → `conda activate yolo` → `python src/xxx.py`
- 桌面有 `YOLO 终端.lnk` 和 `start-yolo.bat` 一键进入环境
- 桌面有用户自己形成的速查表：`C:\Users\RONGG\Desktop\YOLO学习速查.md`

## 目录约定（9-21 重构）
- `data/` 输入素材（图 + 视频，.gitignore 挡着不上传）
- `models/` 权重（yolov8n / yolov8s / yolov8m / yolov8n-pose）
- `src/` 脚本，命名必须一眼看出功能（不用 test1.py 这种）
- `output/` 所有产物；视频帧抽出来放 `output/video_yolo/`
- `学习日志.md` 单文件按天累加；`代码讲解_视频与画框.md` 逐行讲解

## 学习进度
1. ✅ 单图检测（model → results → r.boxes → box.xyxy / cls / conf）
2. ✅ OpenCV 基础：图片=数字表格、`img[行,列]`、BGR、切片左闭右开、resize/imwrite/cvtColor
3. ✅ 用 cv2.rectangle 手画 YOLO 框（已搞懂 `box.xyxy[0]` 剥壳）
4. ✅ 多图批处理 + classes 过滤
5. ✅ 视频抽帧（`%` + `continue`，注意递增写在 continue 前）
6. ✅ 视频逐帧检测存文件（VideoWriter + mp4v + release）
7. ⬜ 视频实时播放（imshow + waitKey + destroyAllWindows）
8. ⬜ 训练自己的模型（LabelImg → data.yaml → model.train）
9. ⬜ 模型导出 ONNX → 边缘部署（RK3588 / Jetson）

## 关键坑（已整理进速查表）
- `mp4` 是容器、`mp4v` 是编码，Windows 自带播放器打不开 mp4v，需换 PotPlayer/VLC
- `imread` 读不到不报错返回 None；`VideoWriter` 目录不存在静默失败
- numpy `img[行,列]` 与 YOLO `(x,y)` 反序
- 张量剥壳：看 `.shape` 有几维就剥几次 `[0]`
