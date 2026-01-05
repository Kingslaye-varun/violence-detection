@echo off
title Violence Detection System
color 0A

:menu
cls
echo ========================================
echo   VIOLENCE DETECTION SYSTEM
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Run Detection (OpenCV)
echo 2. Run Detection with Weapons (YOLO)
echo 3. Run Web Dashboard (Streamlit)
echo 4. Organize Files
echo 5. Install Dependencies
echo 6. Exit
echo.
echo ========================================
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto run_detection
if "%choice%"=="2" goto run_weapon
if "%choice%"=="3" goto run_dashboard
if "%choice%"=="4" goto organize
if "%choice%"=="5" goto install
if "%choice%"=="6" goto exit
goto menu

:run_detection
cls
echo ========================================
echo   Starting Detection System...
echo ========================================
echo.
python run_detection.py
pause
goto menu

:run_weapon
cls
echo ========================================
echo   Starting Weapon Detection System...
echo ========================================
echo.
echo NOTE: First run will download YOLOv8 model
echo.
python weapon_detection_enhanced.py
pause
goto menu

:run_dashboard
cls
echo ========================================
echo   Starting Web Dashboard...
echo ========================================
echo.
echo Opening in browser...
streamlit run streamlit_dashboard_enhanced.py
pause
goto menu

:organize
cls
echo ========================================
echo   Organizing Files...
echo ========================================
echo.
call ORGANIZE.bat
goto menu

:install
cls
echo ========================================
echo   Installing Dependencies...
echo ========================================
echo.
pip install tensorflow opencv-python mediapipe numpy streamlit plotly pandas ultralytics
echo.
echo ========================================
echo   Installation Complete!
echo ========================================
pause
goto menu

:exit
cls
echo.
echo Thank you for using Violence Detection System!
echo.
timeout /t 2 >nul
exit
