# Use a suitable base image.  Python:3.9-slim-buster is a good starting point.
FROM python:3.9-slim-buster

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file (if you have one).  This optimizes layer caching.
COPY requirements.txt .

# Install any Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code.
COPY rent_deploy.py .  # Replace rent_deploy.py with the actual name

# Expose any ports your application uses (if applicable).
# For example, if your app uses Flask on port 5000:
EXPOSE 5000

# Define the command to run when the container starts.
CMD ["python", “rent_deploy.py"] # Replace with how you run your script