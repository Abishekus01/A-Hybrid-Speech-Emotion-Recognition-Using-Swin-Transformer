import os

# Paths to your datasets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESS_PATH = os.path.join(BASE_DIR, "datasets", "TESS")
EMOVO_PATH = os.path.join(BASE_DIR, "datasets", "EMOVO")

# Save model weights here
MODEL_WEIGHTS = os.path.join(BASE_DIR, "model", "model_weights.pth")

# Emotions list
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]