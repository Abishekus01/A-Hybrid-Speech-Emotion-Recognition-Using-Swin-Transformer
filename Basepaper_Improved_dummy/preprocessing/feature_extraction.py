import librosa
import numpy as np
import torch

# =========================
# LOG-MEL SPECTROGRAM
# =========================
def extract_log_mel(file_path, img_size=(128,128)):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = resize_spectrogram(log_mel, img_size)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel)+1e-6)
        return torch.tensor(log_mel, dtype=torch.float32)  # shape: (H,W)
    except:
        return torch.zeros(img_size, dtype=torch.float32)

# =========================
# HANDCRAFTED FEATURES (33-dim)
# =========================
def extract_handcrafted(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = np.mean(chroma.T, axis=0)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast = np.mean(spec_contrast.T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        features = np.hstack([mfcc, chroma, spec_contrast, [zcr], [rms]])  # total 33
        return torch.tensor(features, dtype=torch.float32)
    except:
        return torch.zeros(33, dtype=torch.float32)

# =========================
# RESIZE SPECTROGRAM
# =========================
def resize_spectrogram(spec, img_size):
    h, w = spec.shape
    target_h, target_w = img_size
    resized = np.zeros((target_h, target_w))
    h_scale = h / target_h
    w_scale = w / target_w
    for i in range(target_h):
        for j in range(target_w):
            resized[i,j] = spec[int(i*h_scale), int(j*w_scale)]
    return resized

# =========================
# COMBINED FEATURE EXTRACTION
# =========================
def extract_features(file_path, img_size=(128,128)):
    log_mel = extract_log_mel(file_path, img_size)
    handcrafted = extract_handcrafted(file_path)

    # ADD BATCH & CHANNEL DIMENSIONS
    log_mel = log_mel.unsqueeze(0).unsqueeze(0)  # shape: [1,1,H,W]
    handcrafted = handcrafted.unsqueeze(0)       # shape: [1,33]

    return log_mel, handcrafted