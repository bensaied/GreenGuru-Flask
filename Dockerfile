# Use a base image with Python and system dependencies
FROM python:3.9-slim

# Install system dependencies for OpenCV and other required libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8080

# Set the environment variable for the port
ENV PORT=8080

# Command to run your app
CMD ["python", "app.py"]