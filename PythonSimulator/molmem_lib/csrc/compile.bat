@echo off
setlocal enabledelayedexpansion

:: 1. Search for Visual Studio vcvars64.bat if needed
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    for %%V in ("18\BuildTools" "18\Community" "18\Professional" "18\Enterprise" "2022\BuildTools" "2022\Community" "2022\Professional" "2022\Enterprise" "2019\BuildTools" "2019\Community") do (
        if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\%%~V\VC\Auxiliary\Build\vcvars64.bat" (
            call "%ProgramFiles(x86)%\Microsoft Visual Studio\%%~V\VC\Auxiliary\Build\vcvars64.bat"
            goto :vcvars_found
        )
        if exist "%ProgramFiles%\Microsoft Visual Studio\%%~V\VC\Auxiliary\Build\vcvars64.bat" (
            call "%ProgramFiles%\Microsoft Visual Studio\%%~V\VC\Auxiliary\Build\vcvars64.bat"
            goto :vcvars_found
        )
    )
)
:vcvars_found

:: 2. Auto-detect ROCm / HIP path if not set
if "%HIP_PATH%"=="" (
    if exist "C:\PROGRA~1\AMD\ROCm\7.2" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.2"
    if exist "C:\PROGRA~1\AMD\ROCm\7.1" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.1"
    if exist "C:\PROGRA~1\AMD\ROCm\7.0" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.0"
    if exist "C:\Program Files\AMD\ROCm\7.2" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.2"
    if exist "C:\Program Files\AMD\ROCm\7.1" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.1"
    if exist "C:\Program Files\AMD\ROCm\7.0" set "HIP_PATH=C:\PROGRA~1\AMD\ROCm\7.0"
)
if not "%HIP_PATH%"=="" (
    set "ROCM_HOME=%HIP_PATH%"
    set "ROCM_PATH=%HIP_PATH%"
    set "PATH=%HIP_PATH%\bin;%PATH%"
)

set DISTUTILS_USE_SDK=1
cd /d "%~dp0"

if exist "..\..\molmem-env\Scripts\python.exe" (
    ..\..\molmem-env\Scripts\python.exe setup.py install
) else (
    python setup.py install
)
