 from flask import Flask, request, jsonify
import numpy as np
import cv2
import librosa
import joblib
import tensorflow as tf
import io
import os
import gc
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get(‘GEMINI_API_KEY’)
if GEMINI_API_KEY:
genai.configure(api_key=GEMINI_API_KEY)
chat_model = genai.GenerativeModel(‘gemini-2.5-flash’)
else:
chat_model = None
print(“WARNING: GEMINI_API_KEY not set”, flush=True)

app = Flask(**name**)

# โหลด models ครั้งเดียวตอน startup

# ตัวแปร flag นี้จะเป็น True ก็ต่อเมื่อโมเดลทั้งหมดโหลดเข้า memory เสร็จสมบูรณ์แล้วจริงๆ

# ใช้เช็คผ่าน /health endpoint เพื่อให้แอพมือถือรอจน server พร้อมจริง ไม่ใช่แค่ “ตอบ request ได้”

MODELS_READY = False

print(“Loading TensorFlow Lite model…”, flush=True)
audio_interpreter = tf.lite.Interpreter(model_path=‘jackfruit_model.tflite’)
audio_interpreter.allocate_tensors()
audio_input  = audio_interpreter.get_input_details()
audio_output = audio_interpreter.get_output_details()

print(“Loading surface classifier…”, flush=True)
surface_clf = joblib.load(‘surface_classifier.pkl’)
print(“All models loaded.”, flush=True)

MODELS_READY = True

CLASSES = [‘ขนุนดิบ’, ‘ขนุนสุก’]

SAMPLE_RATE = 22050
N_MFCC      = 40
N_FRAMES    = 100

# — Audio Processing —

def extract_mfcc(audio_bytes):
y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
if mfcc.shape[1] < N_FRAMES:
mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
else:
mfcc = mfcc[:, :N_FRAMES]
result = mfcc[np.newaxis, …, np.newaxis].astype(np.float32)
# คืน memory ของ audio array
del y, mfcc
gc.collect()
return result

def predict_audio(audio_bytes):
mfcc = extract_mfcc(audio_bytes)
audio_interpreter.set_tensor(audio_input[0][‘index’], mfcc)
audio_interpreter.invoke()
result = audio_interpreter.get_tensor(audio_output[0][‘index’])[0]
p_suk = float(result[0])
p_dib = 1.0 - p_suk
print(f”AUDIO RAW: {result}, P(ดิบ)={p_dib:.3f}, P(สุก)={p_suk:.3f}”, flush=True)
del mfcc
gc.collect()
return np.array([p_dib, p_suk], dtype=np.float32)

# — Image Processing —

def white_balance(img):
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
l, a, b = cv2.split(lab)
a = np.clip(a - (np.mean(a) - 128), 0, 255)
b = np.clip(b - (np.mean(b) - 128), 0, 255)
balanced = cv2.merge([l, a, b]).astype(np.uint8)
return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

def extract_surface_features(img_bytes):
nparr = np.frombuffer(img_bytes, np.uint8)
img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
if img is None:
raise ValueError(“อ่านไฟล์รูปภาพไม่ได้ (รูปอาจเสียหายหรือ format ไม่รองรับ)”)
img   = cv2.resize(img, (224, 224))
img   = white_balance(img)

```
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

black_mask  = cv2.inRange(hsv, np.array([0,0,0]),    np.array([180,255,60]))
green_mask  = cv2.inRange(hsv, np.array([25,30,30]), np.array([85,255,255]))
yellow_mask = cv2.inRange(hsv, np.array([15,30,30]), np.array([35,255,255]))

black_ratio  = np.sum(black_mask  > 0) / black_mask.size
green_ratio  = np.sum(green_mask  > 0) / green_mask.size
yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

blurred   = cv2.GaussianBlur(gray, (5,5), 0)
edges     = cv2.Canny(blurred, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spine_count = len([c for c in contours if 10 < cv2.contourArea(c) < 500])
texture     = gray.std()

features = np.array([[black_ratio, green_ratio, yellow_ratio,
                      spine_count, texture]], dtype=np.float32)
# คืน memory ของรูป
del img, hsv, gray, nparr, black_mask, green_mask, yellow_mask, blurred, edges
gc.collect()
return features
```

def predict_image(img_bytes):
features = extract_surface_features(img_bytes)
proba    = surface_clf.predict_proba(features)[0]
del features
gc.collect()
return proba

# — Fusion —

def fuse_predictions(audio_proba, image_proba, audio_weight=0.6, image_weight=0.4):
combined = (audio_weight * audio_proba) + (image_weight * image_proba)
pred_idx = int(np.argmax(combined))
return {
‘result’:      CLASSES[pred_idx],
‘confidence’:  round(float(combined[pred_idx]) * 100, 1),
‘audio_score’: {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(audio_proba)},
‘image_score’: {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(image_proba)},
}

# — Routes —

@app.route(’/’)
def index():
return jsonify({‘status’: ‘ok’, ‘message’: ‘Jackfruit API running’})

@app.route(’/health’)
def health():
“””
เช็คว่าโมเดล AI โหลดเข้า memory เสร็จสมบูรณ์แล้วจริงๆ หรือยัง
ใช้ endpoint นี้ตอน wake-up server จากฝั่งแอพ (ไม่ใช่ ‘/’ เพราะ ‘/’ ตอบได้
เร็วกว่าที่โมเดลโหลดเสร็จจริง ทำให้แอพคิดว่า server พร้อมทั้งที่ยังไม่พร้อม)
“””
if MODELS_READY:
return jsonify({‘status’: ‘ready’, ‘models_loaded’: True}), 200
else:
return jsonify({‘status’: ‘loading’, ‘models_loaded’: False}), 503

@app.route(’/predict’, methods=[‘POST’])
def predict():
if ‘audio’ not in request.files or ‘image’ not in request.files:
return jsonify({‘error’: ‘ต้องส่งทั้ง audio และ image’}), 400
try:
audio_bytes = request.files[‘audio’].read()
image_bytes = request.files[‘image’].read()
audio_proba = predict_audio(audio_bytes)
image_proba = predict_image(image_bytes)
result      = fuse_predictions(audio_proba, image_proba)
return jsonify(result)
except Exception as e:
print(f”PREDICT ERROR: {e}”, flush=True)
gc.collect()
return jsonify({‘error’: f’ประมวลผลไม่สำเร็จ: {str(e)}’}), 500

@app.route(’/predict/audio’, methods=[‘POST’])
def predict_audio_only():
if ‘audio’ not in request.files:
return jsonify({‘error’: ‘ต้องส่ง audio’}), 400
try:
audio_bytes = request.files[‘audio’].read()
proba       = predict_audio(audio_bytes)
pred_idx    = int(np.argmax(proba))
return jsonify({
‘result’:     CLASSES[pred_idx],
‘confidence’: round(float(proba[pred_idx]) * 100, 1),
‘scores’:     {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(proba)}
})
except Exception as e:
print(f”AUDIO ERROR: {e}”, flush=True)
gc.collect()
return jsonify({‘error’: f’ประมวลผลเสียงไม่สำเร็จ: {str(e)}’}), 500

@app.route(’/predict/image’, methods=[‘POST’])
def predict_image_only():
if ‘image’ not in request.files:
return jsonify({‘error’: ‘ต้องส่ง image’}), 400
try:
image_bytes = request.files[‘image’].read()
proba       = predict_image(image_bytes)
pred_idx    = int(np.argmax(proba))
return jsonify({
‘result’:     CLASSES[pred_idx],
‘confidence’: round(float(proba[pred_idx]) * 100, 1),
‘scores’:     {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(proba)}
})
except Exception as e:
print(f”IMAGE ERROR: {e}”, flush=True)
gc.collect()
return jsonify({‘error’: f’ประมวลผลรูปภาพไม่สำเร็จ: {str(e)}’}), 500

@app.route(’/chat’, methods=[‘POST’])
def chat():
if chat_model is None:
return jsonify({‘error’: ‘Chat AI ยังไม่ได้ตั้งค่าบน server’}), 503
data = request.get_json()
if not data or ‘message’ not in data:
return jsonify({‘error’: ‘ต้องส่ง message’}), 400
try:
response = chat_model.generate_content(data[‘message’])
return jsonify({‘reply’: response.text})
except Exception as e:
print(f”CHAT ERROR: {e}”, flush=True)
return jsonify({‘error’: f’เกิดข้อผิดพลาด: {str(e)}’}), 500

if **name** == ‘**main**’:
app.run(host=‘0.0.0.0’, port=5000, debug=False)
