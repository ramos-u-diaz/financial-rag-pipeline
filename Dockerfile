# Start from an official Python image
# This is like saying "start with a clean machine that already has Python 3.11 installed"
# slim means a smaller version without unnecessary extras
FROM python:3.11-slim

# Set the working directory inside the container
# All commands from here on run from this folder
WORKDIR /app

# Copy requirements first — before copying the rest of the code
# This is a Docker best practice for caching
# If your code changes but requirements don't, Docker skips reinstalling libraries
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your code into the container
COPY src/ ./src/
COPY data/ ./data/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
# Notice no --reload flag — that's for development only, not production
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]