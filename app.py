from flask import Flask, request, jsonify
import numpy as np
import cv2
import soundfile as sf
import scipy.signal as signal
import librosa
import tensorflow as tf
import io
import os
import gc
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    chat_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    chat_model = None
    print("WARNING: GEMINI_API_KEY not set", flush=True)

app = Flask(__name__)

MODELS_READY = False

print("Loading audio TFLite model...", flush=True)
audio_interpreter = tf.lite.Interpreter(model_path='jackfruit_model_v2.tflite')
audio_interpreter.allocate_tensors()
audio_input  = audio_interpreter.get_input_details()
audio_output = audio_interpreter.get_output_details()

print("Loading image TFLite model...", flush=True)
image_interpreter = tf.lite.Interpreter(model_path='jackfruit_image_v2.tflite')
image_interpreter.allocate_tensors()
image_input  = image_interpreter.get_input_details()
image_output = image_interpreter.get_output_details()

print("All models loaded.", flush=True)
MODELS_READY = True

CLASSES = ['ขนุนดิบ', 'ขนุนสุก']

SAMPLE_RATE = 22050
N_MFCC      = 40
N_FRAMES    = 100

# --- Audio Processing ---

def extract_mfcc(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_file, dtype='float32')

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != SAMPLE_RATE:
        n_samples = int(len(y) * SAMPLE_RATE / sr)
        y = signal.resample(y, n_samples)

    max_samples = SAMPLE_RATE * 10
    if len(y) > max_samples:
        y = y[:max_samples]

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :N_FRAMES]

    result = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
    del y, mfcc
    gc.collect()
    return result

def predict_audio(audio_bytes):
    mfcc = extract_mfcc(audio_bytes)
    audio_interpreter.set_tensor(audio_input[0]['index'], mfcc)
    audio_interpreter.invoke()
    result = audio_interpreter.get_tensor(audio_output[0]['index'])[0]
    p_suk = float(result[0])
    p_dib = 1.0 - p_suk
    print(f"AUDIO RAW: {result}, P(ดิบ)={p_dib:.3f}, P(สุก)={p_suk:.3f}", flush=True)
    del mfcc
    gc.collect()
    return np.array([p_dib, p_suk], dtype=np.float32)

# --- Image Processing ---

def predict_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("อ่านไฟล์รูปภาพไม่ได้ (รูปอาจเสียหายหรือ format ไม่รองรับ)")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    image_interpreter.set_tensor(image_input[0]['index'], img)
    image_interpreter.invoke()
    result = image_interpreter.get_tensor(image_output[0]['index'])[0][0]
    p_suk = float(result)
    p_dib = 1.0 - p_suk
    print(f"IMAGE RAW: {result}, P(ดิบ)={p_dib:.3f}, P(สุก)={p_suk:.3f}", flush=True)
    del img, nparr
    gc.collect()
    return np.array([p_dib, p_suk], dtype=np.float32)

# --- Fusion ---

def fuse_predictions(audio_proba, image_proba, audio_weight=0.6, image_weight=0.4):
    combined = (audio_weight * audio_proba) + (image_weight * image_proba)
    pred_idx = int(np.argmax(combined))
    return {
        'result':      CLASSES[pred_idx],
        'confidence':  round(float(combined[pred_idx]) * 100, 1),
        'audio_score': {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(audio_proba)},
        'image_score': {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(image_proba)},
    }

# --- Routes ---

@app.route('/')
def index():
    return jsonify({'status': 'ok', 'message': 'Jackfruit API running'})

@app.route('/health')
def health():
    if MODELS_READY:
        return jsonify({'status': 'ready', 'models_loaded': True}), 200
    else:
        return jsonify({'status': 'loading', 'models_loaded': False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'ต้องส่งทั้ง audio และ image'}), 400
    try:
        audio_bytes = request.files['audio'].read()
        image_bytes = request.files['image'].read()
        audio_proba = predict_audio(audio_bytes)
        image_proba = predict_image(image_bytes)
        result      = fuse_predictions(audio_proba, image_proba)
        return jsonify(result)
    except Exception as e:
        print(f"PREDICT ERROR: {e}", flush=True)
        gc.collect()
        return jsonify({'error': f'ประมวลผลไม่สำเร็จ: {str(e)}'}), 500

@app.route('/predict/audio', methods=['POST'])
def predict_audio_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'ต้องส่ง audio'}), 400
    try:
        audio_bytes = request.files['audio'].read()
        proba       = predict_audio(audio_bytes)
        pred_idx    = int(np.argmax(proba))
        return jsonify({
            'result':     CLASSES[pred_idx],
            'confidence': round(float(proba[pred_idx]) * 100, 1),
            'scores':     {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(proba)}
        })
    except Exception as e:
        print(f"AUDIO ERROR: {e}", flush=True)
        gc.collect()
        return jsonify({'error': f'ประมวลผลเสียงไม่สำเร็จ: {str(e)}'}), 500

@app.route('/predict/image', methods=['POST'])
def predict_image_only():
    if 'image' not in request.files:
        return jsonify({'error': 'ต้องส่ง image'}), 400
    try:
        image_bytes = request.files['image'].read()
        proba       = predict_image(image_bytes)
        pred_idx    = int(np.argmax(proba))
        return jsonify({
            'result':     CLASSES[pred_idx],
            'confidence': round(float(proba[pred_idx]) * 100, 1),
            'scores':     {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(proba)}
        })
    except Exception as e:
        print(f"IMAGE ERROR: {e}", flush=True)
        gc.collect()
        return jsonify({'error': f'ประมวลผลรูปภาพไม่สำเร็จ: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if chat_model is None:
        return jsonify({'error': 'Chat AI ยังไม่ได้ตั้งค่าบน server'}), 503
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'ต้องส่ง message'}), 400
    try:
        response = chat_model.generate_content(data['message'])
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"CHAT ERROR: {e}", flush=True)
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
