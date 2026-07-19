# Use an official Python 3.10 base image running on Linux
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this layer is cached by Docker,
# so dependencies are only re-installed when requirements.txt changes)
COPY requirements.txt .

# Install all Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Download the spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
COPY . .

# Create the models directory (it will be populated at runtime from S3)
RUN mkdir -p models

# Expose port 8000 so the container can receive HTTP requests
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]