@echo off
echo Installing Phase 2 - Computer Vision Dependencies for Windows
echo.

echo Updating pip...
python -m pip install --upgrade pip

echo Installing core dependencies...
python -m pip install numpy==1.26.4
python -m pip install opencv-python-headless==4.10.0.84
python -m pip install pillow==10.2.0

echo Installing layout analysis libraries...
python -m pip install layoutparser==0.3.4

echo Installing PyTorch for Windows...
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

echo Installing object detection libraries...
python -m pip install ultralytics==8.1.0

echo Installing utility libraries...
python -m pip install matplotlib==3.8.2
python -m pip install pandas==2.2.0
python -m pip install scipy==1.13.0

echo Installing document processing libraries...
python -m pip install pdf2image==1.16.3
python -m pip install PyMuPDF==1.24.3

echo.
echo All dependencies installed!
echo.
echo Quick verification:
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import layoutparser as lp; print('LayoutParser: OK')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"