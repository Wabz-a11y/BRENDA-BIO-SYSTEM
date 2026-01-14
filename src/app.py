# app.py
from flask import Flask, request, jsonify, send_from_directory
from src.vision import BrendaVision
import threading
import os

app = Flask(__name__, static_folder='.', static_url_path='')
vision = BrendaVision()

@app.route('/')
def index():
    return send_from_directory('.', 'analysis_dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"status": "failed", "error": "No image"}), 400
    return jsonify(vision.analyze_image_b64(data['image']))

@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.json
    if not data or 'name' not in data or 'image' not in data:
        return jsonify({"status": "failed", "error": "Missing name/image"}), 400
    return jsonify(vision.enroll(data['name'], data['image']))

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "yolov8x-pose", "faces": len(vision.emb_db.db)})

@app.route('/voice/identify', methods=['POST'])
def voice_identify():
    # Placeholder — add voice model later
    return jsonify({"user": "Unknown", "confidence": 0.0})

if __name__ == '__main__':
    print("Brenda Vision Server → http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)