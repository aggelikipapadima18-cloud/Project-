@echo off
setlocal

if exist C:\msys64\ucrt64\bin (
    set "PATH=C:\msys64\ucrt64\bin;%PATH%"
)

if not exist build\run_all_assignments_c.exe (
    call build_windows.bat
    if errorlevel 1 exit /b 1
)

build\run_all_assignments_c.exe
