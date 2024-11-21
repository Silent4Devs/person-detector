# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, if needed for API serving)
EXPOSE 80

# Command to run the Python script
CMD ["python", "main.py"]
