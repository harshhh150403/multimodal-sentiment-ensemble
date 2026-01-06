"""
Configuration file for Multimodal Sentiment Analysis project.

Contains all paths, hyperparameters, and constants used across the project.
This centralizes configuration to make the project easier to modify.
"""

import os
import random
import numpy as np
import tensorflow as tf


# ==============================================================================
# PATHS
# ==============================================================================

ROOT_FOLDER = "C:/Users/harsh/OneDrive/Desktop/mp_code/root_folder"
TRANSCRIPT_PATH = os.path.join(ROOT_FOLDER, "Transcript", "Full")
VIDEO_PATH = os.path.join(ROOT_FOLDER, "Video", "Full")
CACHE_DIR = os.path.join(ROOT_FOLDER, "cache")

# Model save paths
MODEL_SAVE_DIR = os.path.join(ROOT_FOLDER, "saved_models")
IFFSA_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "iffsa_model")
BFSA_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "bfsa_model")
TBJE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "tbje_model")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "ensemble_model.joblib")
TOKENIZER_PATH = os.path.join(MODEL_SAVE_DIR, "glove_tokenizer.pkl")


# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

# Video processing
MAX_FRAMES = 100
FRAME_SIZE = 224  # Image size for CNN input (224x224)

# Text processing
MAX_TEXT_LENGTH = 128
VOCAB_SIZE = 10000  # Vocabulary size for GloVe/TBJE model
EMBEDDING_DIM = 100  # Embedding dimension for GloVe

# Training
BATCH_SIZE = 16
EPOCHS = 1  # Increase this for better results (e.g., 10-20)
LEARNING_RATE = 2e-5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3

# Reproducibility
RANDOM_SEED = 42


# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================

def configure_gpu():
    """
    Configure GPU settings for TensorFlow.
    Enables memory growth to prevent OOM errors.
    """
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def set_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: RANDOM_SEED from config)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print("Directories created/verified")
