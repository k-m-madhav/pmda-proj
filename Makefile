.PHONY: help venv install run eval clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "make venv     - create virtualenv in $(VENV)"
	@echo "make install  - install python deps from requirements.txt into $(VENV)"
	@echo "make run      - launch Streamlit app (streamlit_app.py)"
	@echo "make eval     - run validation eval with checkpoints/best_model.pth"
	@echo "make clean    - remove __pycache__"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) -m streamlit run streamlit_app.py

eval: install
	$(PYTHON) -m src.evaluate --checkpoint checkpoints/best_model.pth --split val

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
