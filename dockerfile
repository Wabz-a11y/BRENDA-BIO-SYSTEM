FROM python:3.13-slim

# Install requirements for cv2 not found in python:3.13-slim
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create the user for the application
RUN useradd -m userapp

# Set the working dir and ensure it is owned by our userapp user
WORKDIR /usr/local/app
RUN chown userapp:userapp /usr/local/app

# Install the application dependencies
COPY --chown=userapp:userapp requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY --chown=userapp:userapp src ./src

# Switch to the userapp user
USER userapp

# Epose port 8080
EXPOSE 8080

# Run the application
CMD [ "gunicorn", "--bind", "0.0.0.0:8080", "src.app:app" ]
