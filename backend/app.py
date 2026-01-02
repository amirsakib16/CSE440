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
import sys

# =========================================================
# 0️⃣ TensorFlow MEMORY + THREAD LIMIT (VERY IMPORTANT)
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Limit TensorFlow threads to reduce memory usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Set memory growth for GPU if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"⚠️ GPU config warning: {e}")

# =========================================================
# 1️⃣ Flask App Setup
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")

app = Flask(__name__, static_folder="dist", static_url_path="/")

# Enhanced CORS configuration
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://cse440-nlpqna.onrender.com"
).split(",")

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# =========================================================
# 2️⃣ Load Model & Resources (ONCE)
# =========================================================
print("🔄 Loading model and resources...")

# Check required files exist
required_files = {
    "sentiment_model.h5": "Model file",
    "word_index.json": "Word index",
    "classes.npy": "Class names"
}

missing_files = []
for filename, description in required_files.items():
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        missing_files.append(f"{description} ({filename})")
        print(f"❌ Missing: {filename}")

if missing_files:
    print(f"⚠️ WARNING: Missing files: {', '.join(missing_files)}")
    print("⚠️ Application will start but predictions will fail!")

# Load model
try:
    model_path = os.path.join(BASE_DIR, "sentiment_model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# Load word index
try:
    word_index_path = os.path.join(BASE_DIR, "word_index.json")
    with open(word_index_path, "r") as f:
        word_index = json.load(f)
    print(f"✅ Word index loaded ({len(word_index)} words)")
except Exception as e:
    print(f"❌ Word index error: {e}")
    word_index = {}

# Load class names
try:
    classes_path = os.path.join(BASE_DIR, "classes.npy")
    class_names = np.load(classes_path, allow_pickle=True)
    print(f"✅ Class names loaded ({len(class_names)} classes)")
except Exception as e:
    print(f"❌ Class names error: {e}")
    class_names = []

# Setup NLTK data directory
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Download stopwords with better error handling
try:
    nltk.data.find('corpora/stopwords')
    print("✅ NLTK stopwords already available")
except LookupError:
    try:
        print("⏳ Downloading NLTK stopwords...")
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
        print("✅ NLTK stopwords downloaded")
    except Exception as e:
        print(f"⚠️ NLTK download warning: {e}")

try:
    stop_words = set(stopwords.words("english"))
    print(f"✅ Stopwords loaded ({len(stop_words)} words)")
except Exception as e:
    print(f"⚠️ Stopwords error: {e}, using empty set")
    stop_words = set()

# =========================================================
# 3️⃣ Model Warmup (Prevent Cold Start Issues)
# =========================================================
def warmup_model():
    """Warm up the model to avoid cold start delays"""
    if model is not None and word_index:
        try:
            print("🔥 Warming up model...")
            dummy_text = "This is a test prediction to warm up the model"
            dummy_input = encode_text(dummy_text)
            model.predict(dummy_input, verbose=0)
            print("✅ Model warmed up successfully")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

# =========================================================
# 4️⃣ NLP Helpers
# =========================================================
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)  # Remove HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    
    # Remove stopwords
    if stop_words:
        words = [w for w in text.split() if w not in stop_words]
    else:
        words = text.split()
    
    return " ".join(words)

def encode_text(text: str):
    """Encode text to numerical sequence"""
    if not word_index:
        raise ValueError("Word index not loaded")
    
    tokens = clean_text(text).split()
    seq = [word_index.get(w, 0) for w in tokens]
    return pad_sequences([seq], maxlen=200, padding="post")

# =========================================================
# 5️⃣ API Routes
# =========================================================
@app.route("/api/health")
def health():
    """Enhanced health check endpoint"""
    status = {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "word_index_loaded": bool(word_index),
        "classes_loaded": len(class_names) > 0,
        "stopwords_loaded": bool(stop_words),
        "num_classes": len(class_names) if class_names is not None else 0
    }
    
    http_code = 200 if model is not None else 503
    return jsonify(status), http_code

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs."
        }), 500
    
    if not word_index:
        return jsonify({
            "error": "Word index not loaded. Please check server logs."
        }), 500
    
    if not class_names or len(class_names) == 0:
        return jsonify({
            "error": "Class names not loaded. Please check server logs."
        }), 500

    # Get request data
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({
            "error": "No text provided. Please send JSON with 'text' field."
        }), 400
    
    text = data.get("text", "").strip()
    if not text:
        return jsonify({
            "error": "Empty text provided. Please enter some text."
        }), 400

    # Make prediction
    try:
        x = encode_text(text)
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return jsonify({
            "class": str(class_names[idx]),
            "confidence": f"{confidence:.2%}",  # Format as percentage
            "confidence_raw": confidence,
            "text_length": len(text),
            "processed_tokens": len(clean_text(text).split())
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed. Please try again or contact support."
        }), 500

# =========================================================
# 6️⃣ Serve React Frontend
# =========================================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serve React static files"""
    if not app.static_folder or not os.path.exists(app.static_folder):
        return jsonify({
            "error": "Frontend not built. Run 'npm run build' first."
        }), 404
    
    full_path = os.path.join(app.static_folder, path)
    
    # Serve specific file if it exists
    if path and os.path.exists(full_path) and os.path.isfile(full_path):
        return send_from_directory(app.static_folder, path)
    
    # Otherwise serve index.html (for React Router)
    index_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")
    
    return jsonify({
        "error": "Frontend not found. Please build the React app."
    }), 404

# =========================================================
# 7️⃣ Startup & Shutdown Handlers
# =========================================================
first_request_done = False

@app.before_request
def startup():
    """Run warmup on first request only"""
    global first_request_done
    if not first_request_done:
        warmup_model()
        first_request_done = True

def handle_shutdown(signum, frame):
    """Graceful shutdown handler"""
    print("\n🛑 Shutting down gracefully...")
    sys.exit(0)

# Register signal handlers
import signal
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# =========================================================
# 8️⃣ Local Development Server
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(f"🚀 Starting Flask server on port {port}")
    print(f"{'='*60}\n")
    
    # Warmup in development mode
    warmup_model()
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,  # Keep False for production-like testing
        use_reloader=False  # Avoid double loading
    )