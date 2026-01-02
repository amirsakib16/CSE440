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

# =========================================================
# 0️⃣ TensorFlow MEMORY + THREAD LIMIT (VERY IMPORTANT)
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================================================
# 1️⃣ Flask App Setup
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else None,
    static_url_path=""
)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://cse440amirsakib.onrender.com",
            "http://localhost:5173",  # for local dev
            "http://localhost:4173"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# =========================================================
# 2️⃣ Load Model & Resources (ONCE)
# =========================================================
print("🔄 Loading model and resources...")

try:
    model = tf.keras.models.load_model("sentiment_model.h5", compile=False)
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

try:
    with open("word_index.json", "r") as f:
        word_index = json.load(f)
    print("✅ Word index loaded")
except Exception as e:
    print(f"❌ Word index error: {e}")
    word_index = {}

try:
    class_names = np.load("classes.npy", allow_pickle=True)
    print("✅ Class names loaded")
except Exception as e:
    print(f"❌ Class names error: {e}")
    class_names = []

# Stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# =========================================================
# 3️⃣ NLP Helpers
# =========================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in stop_words)

def encode_text(text: str):
    tokens = clean_text(text).split()
    seq = [word_index.get(w, 0) for w in tokens]
    return pad_sequences([seq], maxlen=200, padding="post")

# =========================================================
# 4️⃣ API Routes
# =========================================================
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

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        x = encode_text(data["text"])
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds))

        return jsonify({
            "class": str(class_names[idx]),
            "confidence": float(np.max(preds))
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# =========================================================
# 5️⃣ Serve React (Optional)
# =========================================================
if app.static_folder:
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path):
        full_path = os.path.join(app.static_folder, path)
        if path and os.path.exists(full_path):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

# =========================================================
# 6️⃣ Local Dev Only (Gunicorn ignores this)
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running locally on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
