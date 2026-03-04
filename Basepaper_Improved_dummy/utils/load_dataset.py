# utils/load_dataset.py

import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from preprocessing.feature_extraction import extract_features
from utils.config import TESS_PATH, EMOVO_PATH, EMOTIONS

# Map first 3 letters of EMOVO filenames to emotions
EMOVO_MAP = {
    "rab": "angry",
    "gio": "happy",
    "pau": "fear",
    "tri": "sad",
    "dis": "disgust",
    "sor": "surprise",
    "neu": "neutral"
}

def get_emotion_from_filename(file, dataset_type):
    """
    Extract emotion label from filename for TESS or EMOVO dataset.
    Handles nested folders in TESS and first-3-letter codes in EMOVO.
    """
    name = file.lower()

    if dataset_type == "TESS":
        # TESS filenames are like oaf_angry.wav inside nested folders
        if "_" in name:
            return name.split("_")[-1].replace(".wav", "")
        else:
            return None

    if dataset_type == "EMOVO":
        # EMOVO filenames are like rab.wav → angry
        code = name[:3]
        return EMOVO_MAP.get(code, None)

    return None

def load_dataset():
    """
    Walk through TESS and EMOVO directories, extract features,
    and return lists: spectrograms, handcrafted features, labels.
    """
    specs = []
    handcrafted_list = []
    labels = []

    # Traverse both datasets
    for dataset_path, dataset_type in [(TESS_PATH, "TESS"), (EMOVO_PATH, "EMOVO")]:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    # Debug print to check files
                    print("Checking file:", file_path)

                    emotion = get_emotion_from_filename(file, dataset_type)

                    if emotion not in EMOTIONS:
                        continue

                    try:
                        # Extract features (log-mel spectrogram + handcrafted)
                        spec, handcrafted = extract_features(file_path)
                        specs.append(spec)
                        handcrafted_list.append(handcrafted)
                        labels.append(EMOTIONS.index(emotion))
                    except Exception as e:
                        print("Skipped:", file_path, "Error:", e)

    print("Total samples:", len(labels))

    if len(labels) == 0:
        raise ValueError("No audio samples found. Check dataset paths.")

    return specs, handcrafted_list, labels