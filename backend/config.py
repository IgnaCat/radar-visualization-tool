import os

class Config:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    ALLOWED_EXTENSIONS = {'nc'}
