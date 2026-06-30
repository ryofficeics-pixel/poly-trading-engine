@echo off
REM Poly Trading Engine v3 - Startup Script
REM =========================================

echo.
echo ========================================
echo  Poly Trading Engine v3
echo  REST-only Auto-Trading Platform
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.8+ or add it to PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists (optional)
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if required files exist
if not exist "ws_server.py" (
    echo [ERROR] ws_server.py not found
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

if not exist ".env" (
    echo [WARNING] .env file not found
    echo Creating default .env configuration...
    (
        echo # Poly Trading Engine Configuration
        echo SKIP_BINANCE_WS=true
        echo FALLBACK_POLL_INTERVAL=5
        echo DEFAULT_CAPITAL=1000.0
        echo MAX_POSITION_PCT=10.0
        echo SIGNAL_THRESHOLD=0.60
        echo TAKE_PROFIT_PCT=1.5
        echo STOP_LOSS_PCT=1.0
        echo MAX_HOLD_MINUTES=60
        echo MAX_OPEN_POSITIONS=3
        echo PORT=8000
    ) > .env
    echo [INFO] Created .env with default settings
)

REM Check if port 8000 is already in use
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8000 is already in use
    echo.
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
        set PID=%%a
        goto :found_pid
    )
    :found_pid
    echo Process ID using port 8000: !PID!
    echo.
    choice /C YN /M "Kill existing process and restart"
    if errorlevel 2 (
        echo [INFO] Keeping existing server running
        goto :open_browser
    )
    echo [INFO] Stopping existing server...
    taskkill /F /PID !PID! >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo [INFO] Starting Poly Trading Engine...
echo [INFO] Server will run on: http://localhost:8000
echo [INFO] Dashboard: http://localhost:8000
echo.
echo [TIP] Press Ctrl+C to stop the server
echo.

REM Start the server
start "Poly Trading Engine" /MIN python ws_server.py

REM Wait for server to start
echo [INFO] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Check if server started successfully
for /L %%i in (1,1,10) do (
    netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
    if not errorlevel 1 (
        echo [SUCCESS] Server started successfully!
        goto :open_browser
    )
    timeout /t 1 /nobreak >nul
)

echo [ERROR] Server failed to start within 10 seconds
echo Check logs/poly_engine.log for details
pause
exit /b 1

:open_browser
echo.
echo [INFO] Opening dashboard in browser...
timeout /t 2 /nobreak >nul
start http://localhost:8000

echo.
echo ========================================
echo  Server Status: RUNNING
echo  Dashboard: http://localhost:8000
echo  Health Check: http://localhost:8000/health
echo ========================================
echo.
echo [INFO] Server is running in background
echo [INFO] To stop: Close the "Poly Trading Engine" window
echo [INFO] Or use: taskkill /F /IM python.exe
echo.
echo Press any key to exit this window (server will keep running)
pause >nul
