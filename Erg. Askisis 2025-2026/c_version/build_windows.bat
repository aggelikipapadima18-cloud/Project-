@echo off
setlocal

if not exist build mkdir build

if exist C:\msys64\ucrt64\bin\gcc.exe (
    set "PATH=C:\msys64\ucrt64\bin;%PATH%"
)

where cmake >nul 2>nul
if %errorlevel%==0 (
    cmake -S . -B build -G "MinGW Makefiles"
    if errorlevel 1 exit /b 1
    cmake --build build
    if errorlevel 1 exit /b 1
    goto done
)

echo CMake was not found. Install CMake/MSYS2 and run this script again.
exit /b 1

:done
echo Build completed. Run from this folder, for example:
echo build\run_all_assignments_c.exe
