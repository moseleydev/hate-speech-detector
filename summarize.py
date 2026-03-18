from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import time
import numpy as np

# Download the NLTK sentence splitter data (only runs once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the API and load the local model into memory
app = FastAPI(
    title="Medical Extractive Summarization API",
    description="A local NLP engine for summarizing medical reports."
)

# Enable CORS so your frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading ML Model... This takes a few seconds.")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

class ReportRequest(BaseModel):
    text: str
    num_sentences: int = 3

@app.post("/api/summarize")
def summarize_report(request: ReportRequest):
    start_time = time.time()
    
    # 1. Tokenize the report into individual sentences
    sentences = nltk.sent_tokenize(request.text)
    
    # Edge case: If the report is shorter than the requested summary length
    if len(sentences) <= request.num_sentences:
        return {"summary": request.text, "metadata": {"status": "too_short"}}
        
    # 2. Convert sentences and the full document into mathematical vectors (embeddings)
    sentence_embeddings = model.encode(sentences)
    doc_embedding = model.encode([request.text])
    
    # 3. Calculate Cosine Similarity
    similarities = cosine_similarity(doc_embedding, sentence_embeddings)[0]
    
    # 4. Rank the sentences by similarity score, then sort chronologically
    ranked_indices = np.argsort(similarities)[::-1]
    top_indices = sorted(ranked_indices[:request.num_sentences])
    
    # 5. Reconstruct the summary
    summary_sentences = [sentences[i] for i in top_indices]
    final_summary = " ".join(summary_sentences)
    
    # Calculate execution time
    process_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        "summary": final_summary,
        "metadata": {
            "processing_time_ms": process_time,
            "original_sentences": len(sentences),
            "summary_sentences": request.num_sentences,
            "compression_ratio": f"{round((request.num_sentences / len(sentences)) * 100)}%"
        }
    }