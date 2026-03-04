from flask import Flask, render_template, request, jsonify
import os

from model.predict import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = "audio/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "ogg", "mp3", "mpeg", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path):
    import librosa
    import soundfile as sf
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    try:
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, y, sr)
        print(f"Converted {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"Conversion error: {e}")
        return input_path

# ───────────────────────────────────────────
# EXISTING ROUTES (no change)
# ───────────────────────────────────────────

@app.route("/")
def home():
    return render_template("emotion_upload.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/purpose")
def purpose():
    return render_template("purpose.html")

# ───────────────────────────────────────────
# EXISTING FILE UPLOAD ROUTE (small fix)
# predict_emotion now returns a dict, so pass it correctly
# ───────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return render_template("emotion_upload.html", error="No audio uploaded")

    audio = request.files["audio"]

    if audio.filename == "":
        return render_template("emotion_upload.html", error="No file selected")

    if not allowed_file(audio.filename):
        return render_template("emotion_upload.html", error="Unsupported file format")

    path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(path)

    # Convert to WAV if needed
    if not path.endswith(".wav"):
        path = convert_to_wav(path)

    emotion_probs = predict_emotion(path)

    if emotion_probs is None:
        return render_template("emotion_upload.html", error="Audio processing failed")

    # Get top emotion
    top_emotion = list(emotion_probs.keys())[0]

    return render_template("emotion_upload.html", emotion=top_emotion, emotion_probs=emotion_probs)

# ───────────────────────────────────────────
# NEW ROUTE — for real-time recorded audio
# ───────────────────────────────────────────

@app.route("/predict-api", methods=["POST"])
def predict_api():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio = request.files["audio"]
    path = os.path.join(UPLOAD_FOLDER, "recorded.webm")
    audio.save(path)
    print(f"Received filename: {audio.filename}")
    print(f"Saved to: {path}")
    print(f"File size: {os.path.getsize(path)} bytes")

    # Convert webm → wav (16kHz mono, as expected by librosa)
    path = convert_to_wav(path)

    emotion_probs = predict_emotion(path)

    if emotion_probs is None:
        return jsonify({"error": "Processing failed"}), 500

    # Get top predicted emotion
    top_emotion = list(emotion_probs.keys())[0]

    return jsonify({
        "emotion": top_emotion,
        "probabilities": emotion_probs
    })

if __name__ == "__main__":
    app.run(debug=True)