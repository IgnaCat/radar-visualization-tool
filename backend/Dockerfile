# Multi-stage build for Python backend with GDAL support
FROM python:3.11-slim as base

# Install system dependencies for GDAL, HDF5, NetCDF4, and other radar libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GDAL and geospatial libraries
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    # HDF5 and NetCDF4 for radar data
    libhdf5-dev \
    libnetcdf-dev \
    # Build tools for Python packages
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    # Utilities
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyART is installed from custom GitHub fork
# Filter out Windows-only packages (pywin32) before installing
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy && \
    GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir GDAL==${GDAL_VERSION} && \
    grep -v "pywin32" requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
# This creates /app/app/ structure which is correct for the package imports
COPY ./app ./app

# Create necessary directories for storage
# Note: these will be overridden by docker-compose volumes in development
RUN mkdir -p /app/storage/uploads \
    /app/storage/tmp \
    /app/storage/data

# Set permissions
RUN chmod -R 755 /app/storage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn
# --log-level info: show request logs and our custom logging
# --timeout-keep-alive 120: prevent premature disconnection during long W operator builds
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--timeout-keep-alive", "120"]
