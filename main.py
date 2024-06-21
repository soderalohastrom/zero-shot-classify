from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from fastapi.encoders import jsonable_encoder
from transformers import pipeline
from typing import Literal

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
        "Politically Undetermined/Unspecified"  # Fallback option
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
        "Lifestyle Undetermined/Unspecified"  # Fallback option
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
        "Child Preference Undetermined/Unspecified"  # Fallback option
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
        "Upbringing Undetermined/Unspecified"  # Fallback option
    ],
    "Geo-Familiarity": [
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
        "Region Undetermined/Unspecified"  # Fallback option
    ]
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import pipeline
    global zeroshot_classifier
    # Load the model from the persistent cache
    zeroshot_classifier = pipeline("zero-shot-classification", 
                                model="MoritzLaurer/bge-m3-zeroshot-v2.0", 
                                cache_dir="/model_cache")
    yield
    del zeroshot_classifier

app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    text: str
    category: Literal["Politics", "Lifestyle", "Children", "Upbringing", "Geo-Familiarity"]
    # hypothesis_template: str = "The given information from this person's dating bio indicates that they most likely fit into the category of {}"
    hypothesis_template: str = "Based on this info from their dating bio, this person is best categorized as {}"

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
        
        predicted_label = zeroshot_classifier(
            text_input.text, 
            subcategories, 
            hypothesis_template=text_input.hypothesis_template
        )
        
        if predicted_label not in subcategories:
            predicted_label = subcategories[-1]
        
        return {"predicted_subcategory": predicted_label}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))