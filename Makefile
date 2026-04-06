# Variables
VENV = torch_env
PYTHON = ./$(VENV)/Scripts/python.exe
PIP = ./$(VENV)/Scripts/pip.exe

.PHONY: help test run setup clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Create venv and install GPU requirements"
	@echo "  make test       - Run the GPU/CUDA verification script"
	@echo "  make run file=X - Run a specific python file (e.g., make run file=train.py)"
	@echo "  make clean      - Remove the virtual environment"

setup:
	python -m venv $(VENV)
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

test:
	$(PYTHON) run_test.py

# Usage: make run file=your_script.py
run:
	$(PYTHON) $(file)

clean:
rm - rf $(VENV)