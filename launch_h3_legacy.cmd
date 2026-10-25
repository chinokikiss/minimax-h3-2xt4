@echo off
setlocal
pushd "%~dp0"
"C:\ProgramData\miniconda3\envs\comfyui\python.exe" -m tools.runtime.launch_h3_legacy %*
set "h3_exit_code=%errorlevel%"
popd
exit /b %h3_exit_code%
