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
# Determine the build folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_build_path = os.path.join(BASE_DIR, '../frontend/build')

# Check if build folder exists
if not os.path.exists(frontend_build_path):
    print(f"⚠️  WARNING: Build folder not found at {frontend_build_path}")
    print("Run 'npm run build' in the frontend directory first!")
    frontend_build_path = None

app = Flask(__name__, 
            static_folder=frontend_build_path if frontend_build_path else None,
            static_url_path='/')

# Configure CORS properly
CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/api/*": {"origins": "*"}
})

# -----------------------------
# 2️⃣ Load Model & Resources
# -----------------------------
print("Loading model and resources...")

try:
    model = tf.keras.models.load_model("sentiment_model.h5")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    with open("word_index.json", "r") as f:
        word_index = json.load(f)
    print("✓ Word index loaded")
except Exception as e:
    print(f"✗ Error loading word_index: {e}")
    word_index = {}

try:
    class_names = np.load("classes.npy", allow_pickle=True)
    print("✓ Class names loaded")
except Exception as e:
    print(f"✗ Error loading classes: {e}")
    class_names = []

# Download stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

# -----------------------------
# 3️⃣ Preprocessing Functions
# -----------------------------
def nlp_clean(text):
    """Clean and preprocess text for the model"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = text.replace('\\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def encode_text(text):
    """Convert text to padded sequence for model"""
    cleaned_text = nlp_clean(text)
    tokens = cleaned_text.split()
    sequence = [word_index[word] for word in tokens if word in word_index]
    padded = pad_sequences([sequence], maxlen=200, padding="post")
    return padded

# -----------------------------
# 4️⃣ API Routes
# -----------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'frontend_served': frontend_build_path is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        processed_input = encode_text(user_text)
        prediction_probs = model.predict(processed_input, verbose=0)
        predicted_index = np.argmax(prediction_probs)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction_probs))
        
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# -----------------------------
# 5️⃣ Serve React Frontend
# -----------------------------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React build files"""
    if frontend_build_path is None:
        return jsonify({
            'error': 'Frontend not built',
            'message': 'Run "npm run build" in the frontend directory'
        }), 404
    
    file_path = os.path.join(app.static_folder, path)
    
    # If file exists, serve it
    if path != "" and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    
    # Otherwise, serve index.html (for React Router)
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    else:
        return jsonify({
            'error': 'index.html not found',
            'build_path': app.static_folder
        }), 404

# -----------------------------
# 6️⃣ Run Flask App
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Flask app on port {port}")
    print(f"📁 Frontend build path: {frontend_build_path}")
    print(f"🔗 API endpoint: http://0.0.0.0:{port}/predict")
    print(f"🌐 Frontend: http://0.0.0.0:{port}/\n")
    
    app.run(
        debug=True,
        port=port,
        host='0.0.0.0'
    )