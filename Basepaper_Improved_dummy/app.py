from flask import Flask, render_template, request
import os
from model.predict import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = "audio/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "ogg", "mp3", "mpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("emotion_upload.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/purpose")
def purpose():
    return render_template("purpose.html")

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

    emotion = predict_emotion(path)

    if emotion is None:
        return render_template("emotion_upload.html", error="Audio processing failed")

    return render_template("emotion_upload.html", emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
