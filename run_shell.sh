#!/bin/bash
# Script to run the MiniPy Shell with the virtual environment activated

cd "$(dirname "$0")"
source venv/bin/activate
python3 mini_shell.py

