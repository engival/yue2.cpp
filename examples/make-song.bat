@echo off
rem Double-click to make the song described in make-song.ps1 (edit that file in Notepad first).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make-song.ps1"
pause
