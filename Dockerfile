FROM python:3.9-slim

# Keep Python output unbuffered so logs stream promptly, and avoid
# writing .pyc files inside the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the pipeline code and its bundled input files.
COPY run.py config.yaml data.csv ./

# No hard-coded paths: run.py always takes its paths from the CLI args
# below, matching the required interface exactly. Files are read/written
# relative to WORKDIR /app inside the container.
CMD ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
