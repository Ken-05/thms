# Use a Python 3.10 base image for the component.
FROM python:3.10-slim-buster

# Set the working directory inside the container.
WORKDIR /end_to_end_automated_pipeline

# Copy the requirements file into the working directory.
COPY requirements.txt .

# Install Python dependencies from the requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main component script into the working directory.
COPY main.py .

# Copy the external training utility scripts.
COPY training_pipeline/ training_pipeline/

# Define the entry point for the container.
ENTRYPOINT ["python", "main.py"]