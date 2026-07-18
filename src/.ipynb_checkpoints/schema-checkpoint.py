from pydantic import BaseModel
from typing import List, Dict

class TextRequest(BaseModel):
    text: str

class FeatureBreakdown(BaseModel):
    lexical_density: float
    nominalization_density: float
    clausal_complexity: float
    avg_clause_length: float
    passive_voice_ratio: float

class ComplexityResponse(BaseModel):
    score: float
    level: str
    features: FeatureBreakdown