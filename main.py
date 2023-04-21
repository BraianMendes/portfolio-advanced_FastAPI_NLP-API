from fastapi import FastAPI
from transformers import pipeline
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("summarization")

@app.post("/sentiment-analysis/")
async def sentiment_analysis(text: str):
    result = sentiment_pipeline(text)
    return {"result": result[0]}


@app.post("/entity-extraction/")
async def entity_extraction(text: str):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
    return {"entities": entities}


@app.post("/summarization/")
async def summarization(text: str):
    result = summarization_pipeline(text)
    return {"summary": result[0]["summary_text"]}
