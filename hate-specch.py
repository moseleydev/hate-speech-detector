from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import re
import time

app = FastAPI(
    title="Toxicity Classification API",
    description="A microservice for detecting hate speech in social media text."
)

# 1. ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontends to connect. You can restrict this to ["http://localhost:3000"] later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Cardiff NLP Twitter RoBERTa model
print("Loading RoBERTa Model... This may take a moment.")
classifier = pipeline(
    "text-classification", 
    model="cardiffnlp/twitter-roberta-base-hate-latest"
)
print("Model loaded successfully!")

class TweetRequest(BaseModel):
    text: str

def preprocess_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '@user', text)
    return text.strip()

@app.post("/api/detect")
def detect_toxicity(request: TweetRequest):
    start_time = time.time()
    
    clean_text = preprocess_tweet(request.text)
    
    # Edge case: If the tweet was only a link and is now empty
    if not clean_text:
        return {"error": "Tweet contains no readable text after preprocessing."}
        
    try:
        # 2. ADD TRUNCATION: Prevents the API from crashing if the text is over 512 tokens
        result = classifier(clean_text, truncation=True, max_length=512)[0]
        
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
    except Exception as e:
        # Catch any unexpected model errors cleanly instead of crashing the server
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")