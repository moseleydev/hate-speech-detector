from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import re
import time

app = FastAPI(
    title="Toxicity Classification API",
    description="A microservice for detecting hate speech in social media text."
)

# Load the Cardiff NLP Twitter RoBERTa model
# This downloads a ~500MB model into memory on the first run
print("Loading RoBERTa Model... This may take a moment.")
classifier = pipeline(
    "text-classification", 
    model="cardiffnlp/twitter-roberta-base-hate-latest"
)
print("Model loaded successfully!")

class TweetRequest(BaseModel):
    text: str

def preprocess_tweet(text: str) -> str:
    # 1. Remove URLs (http/https/www) as they don't contribute to sentiment
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # 2. Standardize handles to '@user' to match the model's training data
    text = re.sub(r'\@\w+', '@user', text)
    # 3. Strip extra whitespace
    return text.strip()

@app.post("/api/detect")
def detect_toxicity(request: TweetRequest):
    start_time = time.time()
    
    # Sanitize the input
    clean_text = preprocess_tweet(request.text)
    
    # Edge case: If the tweet was only a link and is now empty
    if not clean_text:
        return {"error": "Tweet contains no readable text after preprocessing."}
        
    # Run the classification
    # The pipeline returns a list like: [{'label': 'hate', 'score': 0.98}]
    result = classifier(clean_text)[0]
    
    # Clean up the output for the frontend
    label = result['label'].upper()
    confidence = round(result['score'] * 100, 2)
    process_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        "original_tweet": request.text,
        "clean_text": clean_text,
        "prediction": {
            "label": label,  
            "confidence_score": f"{confidence}%",
            "is_toxic": label == "HATE"
        },
        "metadata": {
            "processing_time_ms": process_time,
            "model_used": "cardiffnlp/twitter-roberta-base-hate-latest"
        }
    }