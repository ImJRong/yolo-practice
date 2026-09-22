@echo off
title YOLO Terminal
call "D:\miniconda3\Scripts\activate.bat" D:\miniconda3
call conda activate yolo
cd /d D:\yolo-practice
echo.
echo  ==========================================
echo   [yolo] env ready
echo   dir : D:\yolo-practice
echo   run : python src/detect_image.py
echo  ==========================================
echo.
