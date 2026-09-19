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
from openai import OpenAI
 
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if GROQ_API_KEY:
    chat_model = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
else:
    chat_model = None
    print("WARNING: GROQ_API_KEY not set", flush=True)
 
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')
print(f"Using Groq model: {GROQ_MODEL}", flush=True)
 
MAX_HISTORY = int(os.environ.get('MAX_HISTORY', '10'))
 
app = Flask(__name__)
 
MODELS_READY = False
 
AUDIO_MODEL_PATH = os.environ.get('AUDIO_MODEL_PATH', 'jackfruit_model_v2 (1).tflite')
IMAGE_MODEL_PATH = os.environ.get('IMAGE_MODEL_PATH', 'jackfruit_image_v2.tflite')
 
print("Loading audio TFLite model...", flush=True)
audio_interpreter = tf.lite.Interpreter(model_path=AUDIO_MODEL_PATH)
audio_interpreter.allocate_tensors()
audio_input  = audio_interpreter.get_input_details()
audio_output = audio_interpreter.get_output_details()
print("Audio model loaded.", flush=True)
 
image_interpreter = None
image_input = None
image_output = None
 
print("All models ready.", flush=True)
MODELS_READY = True
 
CLASSES = ['ขนุนดิบ', 'ขนุนสุก']
 
SAMPLE_RATE = 22050
N_MFCC      = 40
N_FRAMES    = 100
 
 
def load_image_model():
    global image_interpreter, image_input, image_output
    if image_interpreter is None:
        print("Loading image TFLite model (lazy)...", flush=True)
        image_interpreter = tf.lite.Interpreter(model_path=IMAGE_MODEL_PATH)
        image_interpreter.allocate_tensors()
        image_input  = image_interpreter.get_input_details()
        image_output = image_interpreter.get_output_details()
        print("Image model loaded.", flush=True)
 
 
# --- OpenCV Surface Analysis ---
 
def analyze_surface(img_bytes):
    """วิเคราะห์ผิวขนุนด้วย OpenCV: สี หนาม ตา texture"""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # 1. สีดำ (ตาขนุน)
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
        black_ratio = np.sum(black_mask > 0) / black_mask.size
 
        # 2. สีเขียว (เปลือก)
        green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
 
        # 3. สีเหลือง (สุก)
        yellow_mask = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([35, 255, 255]))
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
 
        # 4. นับหนาม
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spine_count = len([c for c in contours if 10 < cv2.contourArea(c) < 500])
 
        # 5. texture
        texture = float(gray.std())
 
        # คำนวณ p_suk จาก OpenCV features
        # เหลืองมาก = สุก, เขียวมาก = ดิบ
        p_suk_cv = 0.5
        p_suk_cv += yellow_ratio * 2.0   # เหลืองเพิ่ม p_suk
        p_suk_cv -= green_ratio * 1.5    # เขียวลด p_suk
        p_suk_cv += black_ratio * 0.5    # ตาดำเพิ่ม p_suk นิดหน่อย
        p_suk_cv = float(np.clip(p_suk_cv, 0.0, 1.0))
 
        print(f"OPENCV: yellow={yellow_ratio:.3f}, green={green_ratio:.3f}, "
              f"black={black_ratio:.3f}, spines={spine_count}, "
              f"texture={texture:.1f}, P(สุก)={p_suk_cv:.3f}", flush=True)
 
        return p_suk_cv
 
    except Exception as e:
        print(f"OPENCV ERROR: {e}", flush=True)
        return None
 
 
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
    load_image_model()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("อ่านไฟล์รูปภาพไม่ได้ (รูปอาจเสียหายหรือ format ไม่รองรับ)")
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_float, axis=0)
    image_interpreter.set_tensor(image_input[0]['index'], img_input)
    image_interpreter.invoke()
    result = image_interpreter.get_tensor(image_output[0]['index'])[0][0]
    p_suk_cnn = float(result)
 
    # รวม OpenCV
    p_suk_cv = analyze_surface(img_bytes)
    if p_suk_cv is not None:
        # CNN 70% + OpenCV 30%
        p_suk = p_suk_cnn * 0.7 + p_suk_cv * 0.3
        print(f"IMAGE FUSED: CNN={p_suk_cnn:.3f}, CV={p_suk_cv:.3f}, Final={p_suk:.3f}", flush=True)
    else:
        p_suk = p_suk_cnn
        print(f"IMAGE CNN only: P(สุก)={p_suk:.3f}", flush=True)
 
    p_dib = 1.0 - p_suk
    del img, img_resized, img_rgb, img_float, img_input, nparr
    gc.collect()
    return np.array([p_dib, p_suk], dtype=np.float32)
 
 
# --- Fusion ---
 
def fuse_predictions(audio_proba, image_proba, audio_weight=0.4, image_weight=0.6):
    p_suk_audio = float(audio_proba[1])
    p_suk_image = float(image_proba[1])
 
    if p_suk_image < 0.4 and p_suk_audio > 0.4:
        confidence = round(50 + abs(p_suk_audio - p_suk_image) * 30, 1)
        return {
            'result':      'ขนุนห่าม',
            'confidence':  confidence,
            'audio_score': {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(audio_proba)},
            'image_score': {CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(image_proba)},
        }
 
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
        return jsonify({
            'status': 'ready',
            'models_loaded': True,
            'chat_enabled': chat_model is not None,
            'chat_model': GROQ_MODEL if chat_model is not None else None,
        }), 200
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
  
 
def build_context_prompt(context):
    if not context:
        return None
    result      = context.get('result')
    confidence  = context.get('confidence')
    audio_score = context.get('audio_score')
    image_score = context.get('image_score')
    lines = ["ผลการวิเคราะห์ขนุนล่าสุดของผู้ใช้คนนี้คือ:"]
    if result is not None:
        lines.append(f"- ผลสรุป: {result} (ความมั่นใจ {confidence}%)" if confidence is not None else f"- ผลสรุป: {result}")
    if audio_score:
        score_text = ", ".join(f"{k} {v}%" for k, v in audio_score.items())
        lines.append(f"- จากเสียง: {score_text}")
    if image_score:
        score_text = ", ".join(f"{k} {v}%" for k, v in image_score.items())
        lines.append(f"- จากรูปภาพ: {score_text}")
    return "\n".join(lines)
 
 
def friendly_chat_error(e):
    msg = str(e).lower()
    if 'model_not_found' in msg or 'does not exist' in msg or 'decommissioned' in msg:
        return 'ระบบ AI ขัดข้องชั่วคราว (โมเดลไม่พร้อมใช้งาน) กรุณาแจ้งผู้ดูแลระบบครับ'
    if 'rate_limit' in msg or '429' in msg:
        return 'ตอนนี้มีผู้ใช้งานเยอะ รอสักครู่แล้วลองใหม่นะครับ'
    if 'authentication' in msg or 'api key' in msg or '401' in msg:
        return 'ระบบ AI ยังไม่ได้ตั้งค่า API key ให้ถูกต้อง กรุณาแจ้งผู้ดูแลระบบครับ'
    if 'timeout' in msg or 'timed out' in msg:
        return 'AI ตอบช้าเกินไป ลองถามใหม่อีกครั้งนะครับ'
    return 'ขออภัยครับ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ'
 
 
@app.route('/chat', methods=['POST'])
def chat():
    if chat_model is None:
        return jsonify({'error': 'Chat AI ยังไม่ได้ตั้งค่าบน server'}), 503
    data = request.get_json(silent=True)
    if not data or not data.get('message'):
        return jsonify({'error': 'ต้องส่ง message'}), 400
    user_message = data['message']
    context      = data.get('context')
    history      = data.get('history', []) or []
    try:
        system_context = build_context_prompt(context)
        system_instruction = (
            "คุณคือผู้ช่วย AI ในแอปตรวจสุกขนุน (jackiegem-fruit) "
            "ตอบเป็นภาษาไทย กระชับ เป็นกันเอง และให้ความรู้เกี่ยวกับขนุน "
            "การเลือกขนุนสุก-ดิบ การเก็บรักษา การปรุงอาหาร และอธิบายผลการวิเคราะห์ของแอปได้ "
            "ถ้ามีข้อมูลผลวิเคราะห์ล่าสุดของผู้ใช้ ให้ใช้ข้อมูลนั้นตอบคำถามโดยไม่ต้องถามผู้ใช้ซ้ำ"
        )
        if system_context:
            system_instruction += "\n\n" + system_context
 
        groq_messages = [{'role': 'system', 'content': system_instruction}]
        for h in history[-MAX_HISTORY:]:
            role = h.get('role', 'user')
            role = 'assistant' if role == 'model' else role
            text = h.get('text', '')
            if text:
                groq_messages.append({'role': role, 'content': text})
        groq_messages.append({'role': 'user', 'content': user_message})
 
        completion = chat_model.chat.completions.create(
            model=GROQ_MODEL,
            messages=groq_messages,
            max_tokens=1024,
            timeout=30,
        )
        reply_text = completion.choices[0].message.content
        return jsonify({'reply': reply_text})
    except Exception as e:
        print(f"CHAT ERROR [{GROQ_MODEL}]: {e}", flush=True)
        return jsonify({'error': friendly_chat_error(e)}), 500
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
