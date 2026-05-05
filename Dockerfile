# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install wait-for-it script
ADD https://github.com/vishnubob/wait-for-it/raw/master/wait-for-it.sh /usr/local/bin/wait-for-it
RUN chmod +x /usr/local/bin/wait-for-it

# Copy the requirements.txt file
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the entire project directory
COPY . /app

# Expose ports 8000 (API) and 8501 (Streamlit)
EXPOSE 8000 8501

# Set the entrypoint to use wait-for-it script
ENTRYPOINT ["wait-for-it", "db:5432", "--"]

# Default command (can be overridden in docker-compose.yml)
CMD ["python", "api/run.py"]
