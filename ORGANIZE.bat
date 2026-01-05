@echo off
echo ========================================
echo   ORGANIZING VIOLENCE DETECTION SYSTEM
echo ========================================
echo.

REM Create folders
echo Creating folders...
mkdir models 2>nul
mkdir backup 2>nul
mkdir old_files 2>nul
mkdir docs 2>nul

REM Move models
echo Moving models...
move mobilenet_feature_extractor.tflite models\ 2>nul
move violence_detection_lstm.tflite models\ 2>nul
move gender_classification.tflite models\ 2>nul

REM Move backup models
echo Moving backup files...
move violence_detection_final.h5 backup\ 2>nul
move gender_classification_final.h5 backup\ 2>nul
move training_history.png backup\ 2>nul
move confusion_matrices.png backup\ 2>nul
move simple_stable.zip backup\ 2>nul

REM Move old scripts
echo Moving old scripts...
move combined_detection.py old_files\ 2>nul
move enhanced_detection.py old_files\ 2>nul
move stable_detection.py old_files\ 2>nul

REM Move old docs
echo Moving old docs...
move QUICKSTART.txt old_files\ 2>nul
move FLICKERING_FIX.txt old_files\ 2>nul

REM Move current docs
echo Organizing docs...
move CLEANUP_GUIDE.txt docs\ 2>nul
move START_HERE.txt docs\ 2>nul

echo.
echo ========================================
echo   ORGANIZATION COMPLETE!
echo ========================================
echo.
echo Folder structure:
echo   models/       - Your trained models
echo   backup/       - Backup files
echo   old_files/    - Old versions
echo   docs/         - Documentation
echo.
echo Main files in root:
echo   simple_stable.py
echo   streamlit_dashboard.py
echo   README.md
echo   requirements.txt
echo.
pause
