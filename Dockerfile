# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all of your project files from the repository into the container
COPY . .

# Tell Docker that the container listens on port 7860
# Hugging Face Spaces expects applications to run on this port
EXPOSE 7860

# Define the command to run your app
# This assumes your main file is `main.py` and the FastAPI variable is `app`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
