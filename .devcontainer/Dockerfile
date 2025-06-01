# Use the official Python image from Docker Hub
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies (if needed)
# RUN apt-get update && apt-get install -y <your-dependencies>

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install numpy pandas plotly matplotlib tensorflow scikit-learn keras

# Set the default user (already set to vscode in base image)
USER vscode
