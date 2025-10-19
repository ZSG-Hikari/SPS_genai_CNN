# CNN Image Classifier (FastAPI + PyTorch)

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** and deploys it via **FastAPI** for image classification (CIFAR-10 dataset).  
It can be run using a TOML-based environment (`uv`)


# Quick Start (Using TOML)

# 1 Activate your virtual environment
.venv\Scripts\activate

# 2️ Move into the project directory
cd test\assignment2

# 3 Install dependencies from pyproject.toml
uv sync

# 4 Train your CNN model
uv run python .\main.py

# 5 Run the FastAPI server
uv run uvicorn app.main:app --reload --port 8000

# 6 Open the interactive API docs
Go to 👉 http://127.0.0.1:8000/docs


## Endpoints
GET-> /health -> Health check for the API
POST-> /predict -> Upload an image file (PNG JPG) → returns predicted class and confidence score