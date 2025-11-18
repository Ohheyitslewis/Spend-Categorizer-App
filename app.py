from fastapi import FastAPI
from pydantic import BaseModel
from classifier import load_taxonomy, classify_with_ollama

app = FastAPI(title="Item Description Classifier")

# Load once at startup
TAXONOMY = load_taxonomy("taxonomy.json")

class ClassifyIn(BaseModel):
    description: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/classify")
def classify(inp: ClassifyIn):
    result = classify_with_ollama(inp.description, TAXONOMY, model_name="llama3")
    return result.model_dump()


