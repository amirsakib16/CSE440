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
# 1️⃣ BASE DIRECTORY (MUST BE FIRST)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================================================
# 2️⃣ NLTK OFFLINE-SAFE SETUP
# =========================================================
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.data.find("corpora/stopwords")
    print("✅ NLTK stopwords already available")
except LookupError:
    raise RuntimeError("NLTK stopwords not found. Pre-download and include in repo.")
# Load stopwords ONCE
stop_words = set(stopwords.words("english"))
# =========================================================
# 3️⃣ TensorFlow MEMORY + THREAD LIMIT
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# =========================================================
# 4️⃣ Flask App Setup
# =========================================================
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")
app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app)
# =========================================================
# 5️⃣ Load Model & Resources (ONCE)
# =========================================================
print("🔄 Loading model and resources...")
try:
    model_path = os.path.join(BASE_DIR, "sentiment_model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Model loaded from {model_path}")
except Exception as e:
    print(f"❌ Model load error: {e} (path: {model_path})")
    model = None
try:
    word_index_path = os.path.join(BASE_DIR, "word_index.json")
    with open(word_index_path, "r") as f:
        word_index = json.load(f)
    print(f"✅ Word index loaded from {word_index_path}")
except Exception as e:
    print(f"❌ Word index error: {e} (path: {word_index_path})")
    word_index = {}
try:
    classes_path = os.path.join(BASE_DIR, "classes.npy")
    class_names = np.load(classes_path, allow_pickle=True)
    print(f"✅ Class names loaded from {classes_path}")
except Exception as e:
    print(f"❌ Class names error: {e} (path: {classes_path})")
    class_names = []
# =========================================================
# 6️⃣ NLP Helpers
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
# 7️⃣ API Routes
# =========================================================
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })
@app.route("/predict", methods=["POST"])
def predict():
    print("Received predict request")
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
# 8️⃣ Serve React
# =========================================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")
# =========================================================
# 9️⃣ Local Dev Only
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running locally on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)