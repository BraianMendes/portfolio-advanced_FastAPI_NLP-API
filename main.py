from fastapi import FastAPI, Request
from transformers import pipeline
import spacy
from pydantic import BaseModel, Field
from typing import Optional
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi import Limiter

app = FastAPI()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")
summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

# Modelos de entrada para cada endpoint
class SentimentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class NerInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class SummarizationInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=1000)
    min_length: Optional[int] = Field(30, ge=10)
    max_length: Optional[int] = Field(100, ge=10)


@app.post("/sentiment-analysis/")
@limiter.limit("5/minute")
async def sentiment_analysis(request: Request, input_data: SentimentInput):
    result = sentiment_pipeline(input_data.text)
    return {"result": result[0]}

@app.post("/entity-extraction/")
@limiter.limit("5/minute")
async def entity_extraction(request: Request, input_data: NerInput):
    doc = nlp(input_data.text)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
    return {"entities": entities}

@app.post("/summarization/")
@limiter.limit("2/minute")
async def summarization(request: Request, input_data: SummarizationInput):
    result = summarization_pipeline(input_data.text, min_length=input_data.min_length, max_length=input_data.max_length)
    return {"summary": result[0]["summary_text"]}