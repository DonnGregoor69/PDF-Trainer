@echo off

REM Build package 
python setup.py develop

REM Run main script
python main.py

REM Run tests
python -m unittest discover

REM Rebuild package 
python setup.py develop

pause