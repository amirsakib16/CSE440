from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import os

# -----------------------------
# 1️⃣ Flask App Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ VITE BUILD PATH (VERY IMPORTANT)
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")


if not os.path.exists(FRONTEND_DIST):
    print(f"⚠️  WARNING: Frontend build not found at {FRONTEND_DIST}")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIST,
    static_url_path=""
)

CORS(app)

# -----------------------------
# 2️⃣ Load Model & Resources
# -----------------------------
print("Loading model and resources...")

try:
    model = tf.keras.models.load_model("sentiment_model.h5")
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Model load error: {e}")
    model = None

try:
    with open("word_index.json", "r") as f:
        word_index = json.load(f)
    print("✓ Word index loaded")
except Exception as e:
    print(f"✗ Word index error: {e}")
    word_index = {}

try:
    class_names = np.load("classes.npy", allow_pickle=True)
    print("✓ Class names loaded")
except Exception as e:
    print(f"✗ Class names error: {e}")
    class_names = []

# Stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -----------------------------
# 3️⃣ NLP Helpers
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return " ".join(w for w in words if w not in stop_words)

def encode_text(text):
    tokens = clean_text(text).split()
    seq = [word_index[w] for w in tokens if w in word_index]
    return pad_sequences([seq], maxlen=200, padding="post")

# -----------------------------
# 4️⃣ API Routes
# -----------------------------
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    x = encode_text(text)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds))

    return jsonify({
        "class": str(class_names[idx]),
        "confidence": float(np.max(preds))
    })

# -----------------------------
# 5️⃣ Serve React (Vite)
# -----------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# -----------------------------
# 6️⃣ Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on port {port}")
    print(f"📁 Frontend path: {FRONTEND_DIST}")
    app.run(host="0.0.0.0", port=port)
