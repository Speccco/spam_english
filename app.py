import os
import re
import pickle
import requests
import warnings
import logging
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# Setup
# ===============================
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ===============================
# English Profanity Tester Class
# ===============================
class EnglishProfanityTester:
    def __init__(self, repo_id='TheMajic/spam_english', max_length=25):
        """Initialize the tester with the model from Hugging Face Hub."""
        self.repo_id = repo_id
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        self._load_model_and_tokenizer()

    def _download_file(self, url, filename):
        """Downloads a file if it doesn't exist."""
        if not os.path.exists(filename):
            logger.info(f"🔄 Downloading {filename} from {url}...")
            try:
                response = requests.get(url, allow_redirects=True)
                response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(response.content)
                logger.info(f"✅ {filename} downloaded successfully.")
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Failed to download {filename}. Error: {e}")
                raise

    def _load_model_and_tokenizer(self):
        """Loads the Keras model and tokenizer."""
        repo_url = f"https://huggingface.co/{self.repo_id}/resolve/main/"
        model_filename = "best_spam_model.keras"
        tokenizer_filename = "tokenizer.pkl"

        # Download files
        self._download_file(repo_url + model_filename, model_filename)
        self._download_file(repo_url + tokenizer_filename, tokenizer_filename)

        # Load model
        try:
            self.model = tf.keras.models.load_model(model_filename)
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

        # Load tokenizer
        try:
            with open(tokenizer_filename, "rb") as f:
                self.tokenizer = pickle.load(f)
            logger.info("✅ Tokenizer loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading tokenizer: {e}")
            raise

    def preprocess_text(self, text):
        """Simple text preprocessing for English."""
        if not isinstance(text, str):
            return ""
        
        # Remove Arabic characters
        text = re.sub(r'[\u0600-\u06FF]+', '', text)
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def check_bad_words(self, text):
        """Check if text contains explicit English bad words."""
        bad_words = [
            'fuck', 'bitch', 'shit', 'asshole', 'cunt', 'dick', 
            'pussy', 'bastard', 'motherfucker', 'cock'
        ]
        
        text_lower = text.lower()
        found_words = [word for word in bad_words if word in text_lower.split()]
        
        return len(found_words) > 0, found_words

    def predict(self, text):
        """Predict if text is profane with a bad words override."""
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'original_text': text,
                'final_prediction': 'Good',
                'final_class': 0,
                'override_reason': 'Input was empty after preprocessing.'
            }

        has_bad_words, found_bad_words = self.check_bad_words(processed_text)
        
        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Model prediction
        model_prob = self.model.predict(padded_sequences, verbose=0)[0][0]
        model_prediction_class = 1 if model_prob >= 0.5 else 0
        
        # Final decision with override
        if has_bad_words:
            final_prediction = "Bad"
            final_class = 1
            override_reason = f"Contains explicit bad words: {', '.join(found_bad_words)}"
        else:
            final_prediction = "Bad" if model_prediction_class == 1 else "Good"
            final_class = model_prediction_class
            override_reason = None
            
        return {
            'original_text': text,
            'processed_text': processed_text,
            'model_prediction': 'Profane' if model_prediction_class == 1 else 'Not Profane',
            'model_confidence': float(model_prob),
            'final_prediction': final_prediction,
            'final_class': final_class,
            'has_bad_words': has_bad_words,
            'found_bad_words': found_bad_words,
            'override_reason': override_reason
        }

# ===============================
# FastAPI Application
# ===============================

class ProfanityRequest(BaseModel):
    text: str

class BatchProfanityRequest(BaseModel):
    texts: list[str]

app = FastAPI(
    title="English Profanity Filter API",
    description="An API to detect profanity in English text using a Keras model with a rule-based override.",
    version="1.1.0"
)

tester = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global tester
    try:
        tester = EnglishProfanityTester()
        logger.info("🚀 English Profanity Filter API is ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model on startup: {e}")
        raise

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the English Profanity Filter API"}

@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy" if tester else "unhealthy"}

@app.post("/predict", tags=["Prediction"])
async def predict_profanity(request: ProfanityRequest):
    """Predicts if the given English text contains profanity."""
    if not tester:
        return {"error": "Model not loaded"}
    
    try:
        return tester.predict(request.text)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/batch", tags=["Prediction"])
async def predict_batch_profanity(request: BatchProfanityRequest):
    """Predicts profanity for a batch of English texts."""
    if not tester:
        return {"error": "Model not loaded"}
        
    try:
        results = [tester.predict(text) for text in request.texts]
        return {
            "predictions": results,
            "summary": {
                "total": len(results),
                "bad_count": sum(1 for r in results if r['final_prediction'] == 'Bad'),
                "good_count": sum(1 for r in results if r['final_prediction'] == 'Good'),
            }
        }
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}
