from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from model import SentimentAnalyser
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A REST API for analyzing sentiment of text reviews",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
try:
    analyzer = SentimentAnalyser()
except Exception as e:
    print(f"Error loading models: {e}")
    analyzer = None

# Pydantic models for request/response
class ReviewRequest(BaseModel):
    text: str
    review_id: Optional[str] = None

class SentimentResponse(BaseModel):
    review_id: Optional[str] = None
    text: str
    sentiment: str
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API status"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=analyzer is not None
    )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: ReviewRequest):
    """
    Analyze the sentiment of a given text review.
    
    Args:
        request: Contains the text to analyze and optional review ID
        
    Returns:
        SentimentResponse with sentiment classification
    """
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Sentiment analyzer not available. Models failed to load."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text field cannot be empty"
        )
    
    try:
        sentiment = analyzer.predict(request.text)
        
        return SentimentResponse(
            review_id=request.review_id,
            text=request.text,
            sentiment=sentiment
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@app.get("/analyze/text/{text}")
async def analyze_sentiment_text(text: str):
    """
    Alternative endpoint to analyze sentiment via URL path.
    
    Args:
        text: The text to analyze
        
    Returns:
        SentimentResponse with sentiment classification
    """
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Sentiment analyzer not available. Models failed to load."
        )
    
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        sentiment = analyzer.predict(text)
        
        return SentimentResponse(
            text=text,
            sentiment=sentiment
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

