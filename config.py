import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Constants
CONDITIONS = ['acne', 'blackheads', 'dark spots', 'dryness', 'normal', 'oily', 'pores', 'wrinkles']
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join(MODELS_DIR, "fixed_skin_model.pth")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "thresholds.json")

# Backend Constants
API_BASE_URL = "http://localhost:5000/api"
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
