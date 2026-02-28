import torch
import torch.nn.functional as F
from model.swin_tser import SwinSERModel
from preprocessing.feature_extraction import extract_features
from utils.config import MODEL_WEIGHTS, EMOTIONS

device = torch.device("cpu")

# Initialize model with correct handcrafted_dim
model = SwinSERModel(num_classes=len(EMOTIONS), handcrafted_dim=33).to(device)

# Load checkpoint but ignore handcrafted_fc (size mismatch)
checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
model_dict = model.state_dict()

# Keep only keys compatible with current model
filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)

# Load updated state dict
model.load_state_dict(model_dict)
model.eval()

def predict_emotion(file_path):
    try:
        log_mel, handcrafted = extract_features(file_path)

        # Ensure handcrafted dimension is correct
        if handcrafted.size(1) != 33:
            handcrafted = handcrafted[:, :33]

        with torch.no_grad():
            log_mel = log_mel.to(device)
            handcrafted = handcrafted.float().to(device)
            output = model(log_mel, handcrafted)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        # Create dict of emotion -> probability %
        emotion_probs = {EMOTIONS[i]: round(float(probs[i])*100,2) for i in range(len(EMOTIONS))}

        # Sort descending
        emotion_probs = dict(sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True))

        return emotion_probs

    except Exception as e:
        print("Error in prediction:", e)
        return None