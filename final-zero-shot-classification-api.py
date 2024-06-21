from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from fastapi.encoders import jsonable_encoder
from typing import Literal

# Updated categories with fallback options
CATEGORIES = {
    "Politics": [
        "Conservative Traditionalist",
        "Progressive Activist",
        "Moderate Pragmatist",
        "Libertarian Thinker",
        "Apolitical Minimalist",
        "Eco-Conscious Liberal",
        "Independent Swing Voter",
        "Cultural Conservative",
        "Liberal Intellectual",
        "Non-Political Humanist",
        "Fiscally Conservative, Socially Liberal",
        "Politically Flexible Pragmatist",
        "Politically Unspecified"  # Fallback option
    ],
    "Lifestyle": [
        "Adventurous Travelers",
        "Health Enthusiasts",
        "Social Butterflies",
        "Fitness Fanatics",
        "Culinary Explorers",
        "Family and Community Oriented",
        "Intellectuals and Lifelong Learners",
        "Balanced Work-Life Advocates",
        "Homebodies with Hobbies",
        "Active and Outdoorsy",
        "Spiritual and Mindful",
        "Cultural Connoisseurs",
        "Lifestyle Undetermined"  # Fallback option
    ],
    "Children": [
        "No Children, Wants Children",
        "No Children, Open to Having Children",
        "No Children, Does Not Want Children",
        "Has Children, Open to More",
        "Has Children, Does Not Want More",
        "No Preference for Having Children",
        "Specific Conditions for Having/Accepting Children",
        "Has Children, Specific Preferences for New Children",
        "Adoptive Parents",
        "No Children, Prefers Not to Have Children",
        "No Children, Unsure About Having Children",
        "Has Adult Children, No Interest in Having More",
        "No Children, Prefers No Kids",
        "Child Preference Unspecified"  # Fallback option
    ],
    "Upbringing": [
        "Traditional Family Structure",
        "Single Parent Household",
        "Blended Family",
        "Adoptive or Foster Family",
        "Multicultural Background",
        "Religious Upbringing",
        "Secular Upbringing",
        "Rural Upbringing",
        "Urban Upbringing",
        "Military Family Background",
        "Immigrant Family Background",
        "Academic-Focused Upbringing",
        "Upbringing Unspecified"  # Fallback option
    ],
    "Geo-regional Familiarity": [
        "California",
        "Florida",
        "Texas & Deep South",
        "Northeast Corridor",
        "New England",
        "Atlantic Coast & Appalachia",
        "Great Lakes & Midwest",
        "Mountain West",
        "Southwest Desert",
        "Cascadia & Alaska",
        "Canada",
        "Mexico & South America",
        "Western Europe",
        "Australia & Pacific Island",
        "Africa & Caribbean",
        "Asia & Baltics",
        "Region Unspecified"  # Fallback option
    ]
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import pipeline
    global zeroshot_classifier
    zeroshot_classifier = pipeline("zero-shot-classification", 
                                   model="MoritzLaurer/bge-m3-zeroshot-v2.0", 
                                   cache_dir="./model_cache")
    yield
    del zeroshot_classifier

app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    text: str
    category: Literal["Politics", "Lifestyle", "Children", "Upbringing", "Geo-regional Familiarity"]
    hypothesis_template: str = "This person's dating bio indicates they {}"

@app.get('/')
async def welcome():
    return "Welcome to our Dating Bio Classification API"

MAX_TEXT_LENGTH = 1000

@app.post('/analyze')
async def classify_text(text_input: TextInput):    
    try:
        text_input_dict = jsonable_encoder(text_input)
        text_data = TextInput(**text_input_dict)

        if len(text_input.text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail="Text length exceeds maximum allowed length")
        elif len(text_input.text) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        subcategories = CATEGORIES[text_input.category]
        
        result = zeroshot_classifier(
            text_input.text, 
            subcategories, 
            hypothesis_template=text_input.hypothesis_template,
            multi_label=False
        )
        
        result["main_category"] = text_input.category
        
        if result['scores'][0] < 0.3:
            result['labels'][0] = subcategories[-1]
            result['scores'][0] = 1.0
        
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
