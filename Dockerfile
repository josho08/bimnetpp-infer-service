# … earlier steps installing dependencies …

# Copy in your service code and weights
COPY main.py .
COPY BIM-Net++_HePIC.pth .

# Expose nothing (serverless uses /run internally), but just to be safe:
EXPOSE 8000

# Start the serverless handler
CMD ["python", "main.py"]
