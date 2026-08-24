# C++ / OpenCV assignments

These files provide one native C++ / OpenCV executable for each assignment.

Build with CMake:

```powershell
cd "Erg. Askisis 2025-2026\c_version"
cmake -S . -B build
cmake --build build
```

Or build on Windows with the included script:

```powershell
cd "Erg. Askisis 2025-2026\c_version"
.\build_windows.bat
```

The script automatically uses the MSYS2 compiler from `C:\msys64\ucrt64\bin` when it exists.

Run one exercise:

```powershell
.\build\askisi1_erotima1.exe
```

Run all exercises:

```powershell
.\build\run_all_assignments_c.exe
```

If you run from normal PowerShell, use this script so the OpenCV DLL path is available:

```powershell
.\run_all_cpp.bat
```
