# # English Profanity Filter API

A high-performance FastAPI application that uses a Keras/TensorFlow model to detect profanity in English text. This API includes a rule-based override to filter explicit bad words and remove non-English characters.

The model is hosted on Hugging Face at [TheMajic/spam_english](https://huggingface.co/TheMajic/spam_english).

---

## Key Features

-   **AI-Powered Classification**: Predicts if text is profane using a fine-tuned model.
-   **Rule-Based Filtering**: Instantly flags text containing explicit bad words.
-   **Text Cleaning**: Removes URLs, mentions, and Arabic characters before analysis.
-   **Batch Processing**: Use the `/batch` endpoint to analyze multiple texts at once.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

---

## API Usage Example

### Predict a Single Text

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "This is some example text."}'
```

### Predict a Batch of Texts

```bash
curl -X POST "http://localhost:8000/batch" \
-H "Content-Type: "application/json" \
-d '{"texts": ["A clean sentence.", "A fucking dirty one."]}'
```