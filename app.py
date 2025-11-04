import os
import sys
import pickle
import requests
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# إعداد البيئة
# ===============================

# تعطيل GPU لتجنب مشاكل CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# تقليل تحذيرات TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# تأكد أن الإخراج يستخدم UTF-8 (لتفادي مشاكل الإيموجي)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ===============================
# إعداد روابط الملفات
# ===============================

repo_url = "https://huggingface.co/Ahmad1020/sarcasm-spam-detector/resolve/main/"
model_filename = "best_spam_model.keras"
tokenizer_filename = "tokenizer.pkl"
max_length = 25  # يجب أن يكون نفس max_length المستخدم أثناء التدريب

# ===============================
# تحميل الملفات من Hugging Face
# ===============================

# تحميل النموذج إذا لم يكن موجودًا
if not os.path.exists(model_filename):
    print(f"Downloading {model_filename}...")
    model_response = requests.get(repo_url + model_filename)
    if model_response.status_code == 200:
        with open(model_filename, "wb") as f:
            f.write(model_response.content)
    else:
        raise Exception(f"Failed to download {model_filename}. Status code: {model_response.status_code}")

# تحميل التوكنيزر إذا لم يكن موجودًا
if not os.path.exists(tokenizer_filename):
    print(f"Downloading {tokenizer_filename}...")
    tokenizer_response = requests.get(repo_url + tokenizer_filename)
    if tokenizer_response.status_code == 200:
        with open(tokenizer_filename, "wb") as f:
            f.write(tokenizer_response.content)
    else:
        raise Exception(f"Failed to download {tokenizer_filename}. Status code: {tokenizer_response.status_code}")

# ===============================
# تحميل النموذج
# ===============================

try:
    model = tf.keras.models.load_model(model_filename)
    print("[OK] Model loaded successfully!")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# ===============================
# تحميل التوكنيزر
# ===============================

try:
    with open(tokenizer_filename, "rb") as f:
        tokenizer = pickle.load(f)
    print("[OK] Tokenizer loaded successfully!")
except Exception as e:
    raise Exception(f"Error loading tokenizer: {e}")

# ===============================
# دالة التنبؤ
# ===============================

def predict_sarcasm(sentence: str) -> bool:
    """
    ترجع True إذا كانت الرسالة سخرية (Spam/Sarcasm)
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return False  # تجاهل الإدخال الفارغ أو غير النصي

    # تحويل النص إلى تسلسل
    test_sequences = tokenizer.texts_to_sequences([sentence])
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    # التنبؤ
    pred = model.predict(test_review_pad, verbose=0)[0][0] * 100
    return pred >= 50  # يعتبر سخرية إذا ≥ 50%

# ===============================
# إنشاء FastAPI
# ===============================

app = FastAPI(title="Spam/Sarcasm Detector API", version="1.0")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    """
    إدخال نص → إرجاع إذا كان سخرية (True/False)
    """
    result = predict_sarcasm(data.text)
    return {"is_sarcastic": result}
