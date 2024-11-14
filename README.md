# FastAPI Application with Gemini Pro Model and MongoDB

This project is a FastAPI application that uses the Gemini Pro model to answer questions based on embeddings stored in a MongoDB collection. The application entry point is `main.py`, and it can be run locally with the following steps.

## Prerequisites

- Python 3.x
- MongoDB instance (local or cloud-based)
- Virtual environment (recommended)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   
2. **Run command**:
   ```bash
   python -m venv .venv
   
4. **Run command**:
   ```bash
   .\.venv\Scripts\activate
   
5. **Run command**:
   ```bash
   pip install -r requirements.txt
   
6. **Run command**:
   ```bash
   uvicorn main:app --reload

7. **Run command to deactivate**:
   ```bash
   deactivate
