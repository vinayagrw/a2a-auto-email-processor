@echo off
setlocal enabledelayedexpansion

:: Ports used by the agents
set "ports=8001 8002 8003"

echo Searching for agent processes...

for %%p in (%ports%) do (
    echo Checking port %%p...
    
    :: Find process using the port
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p "') do (
        set "pid=%%a"
        
        :: Get process info
        for /f "tokens=1,2 delims=," %%b in ('tasklist /fi "PID eq !pid!" /nh /fo:csv') do (
            set "name=%%~b"
            set "name=!name:"=!"
            echo Found process using port %%p:
            echo   PID: !pid!
            echo   Name: !name!
            
            :: Kill the process
            taskkill /F /PID !pid! >nul 2>&1
            if !errorlevel! equ 0 (
                echo   Successfully terminated process
            ) else (
                echo   Failed to terminate process
            )
        )
    )
    
    if not defined pid (
        echo No process found using port %%p
    )
    
    echo ----------------------------------------
)

echo Done!
endlocal