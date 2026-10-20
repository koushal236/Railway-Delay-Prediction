@echo off
setlocal
cd /d "%~dp0"
set "PATH=C:\Users\bhava\AppData\Local\Programs\Git\cmd;C:\Users\bhava\AppData\Local\Programs\Git\ucrt64\bin;%PATH%"

echo ======================================================================
echo Railway Delay Prediction ML Capstone - GitHub Push
echo Target: https://github.com/koushal236/Railway-Delay-Prediction.git
echo Branch: main
echo Commit: eafcdd9
echo ======================================================================
echo.
echo Running git push origin main...
echo (If a GitHub sign-in window appears in your browser, please authorize it.)
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo SUCCESS: Project pushed successfully to GitHub!
    echo View repository: https://github.com/koushal236/Railway-Delay-Prediction
    echo =======================================================================
) else (
    echo.
    echo ======================================================================
    echo Push encountered an issue. See error details above.
    echo ======================================================================
)

echo.
pause
