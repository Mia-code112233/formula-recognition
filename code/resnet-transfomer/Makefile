.PHONY: help
help:
	@echo "Commands:"

	@echo "venv               : sets up virtual environment for development."
	@echo "api                : launches FastAPI app."
	@echo "docker             : builds and runs a docker image."
	@echo "streamlit          : runs streamlit app."
	@echo "lint               : runs linting."
	@echo "clean              : cleans all unnecessary files."


# Set up virtual environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-dev


# API
.PHONY: api
api:
	uvicorn api.app:app --host 0.0.0.0 --port 9999 --reload --reload-dir models --reload-dir api

# Docker
.PHONY: docker
docker:
	docker build -t image-to-latex:latest -f Dockerfile .
	docker run -p 8000:8000 --name image-to-latex image-to-latex:latest

# Streamlit
.PHONY: streamlit
streamlit:
	streamlit run streamlit/app.py

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . -type f -name ".coverage*" -ls -delete
	rm -rf htmlcov
	rm -rf .mypy_cache
