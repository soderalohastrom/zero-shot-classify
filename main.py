from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from fastapi.encoders import jsonable_encoder
from transformers import pipeline
from typing import Literal, List, Dict
import re
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "Politically Undetermined"  # Fallback option
    ],
    "Lifestyle": [
        "Adventurous Traveler",
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
        "Wellness and Self-Care Advocates",
        "Social and Professional Networkers",
        "Community Builders and Volunteers",
        "Nature Lovers and Environmentalists",
        "Simplifiers and Decluttering Enthusiasts",
        "Philanthropists and Humanitarians",
        "Fashion and Style Enthusiasts",
        "Pet Owners and Animal Lovers",
        "Adrenaline Junkies and Thrill Seekers",
        "Collectors and Antique Enthusiasts",
        "Sustainable Living Advocates",
        "Career Achievers and Go-Getters",
        "Luxury and High-End Lifestyle Enthusiasts",
        "Free Spirits and Nonconformists",
        "Classic and Timeless Style Aficionados",
        "Country Living and Rustic Enthusiasts",
        "City Dwellers and Urban Explorers",
        "Digital Nomads and Remote Workers",
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
        "Child Preference Undetermined"  # Fallback option
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
        "Upbringing Undetermined"  # Fallback option
    ],
    "Geo-Familiarity": [
        "California",
        "Florida",
        "Texas & Deep South",
        "NYC & DC to Boston Corridor",
        "Northern New England",
        "Atlantic Coast & Appalachia",
        "Great Lakes & Midwest",
        "Mountain West and Prairie",
        "Southwest Desert",
        "Cascadia & Alaska",
        "Canada",
        "Mexico & South America",
        "Western Europe",
        "Middle East & North Africa",
        "Australia & Pacific Island",
        "Tropical Africa & Caribbean",
        "Asia & The Baltics",
        "Region Undetermined"  # Fallback option
    ]
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import pipeline, AutoTokenizer
    global zeroshot_classifier
    logger.info("Loading zero-shot classification model...")
    start_time = time.time()
    
    # Get the model's maximum sequence length
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", cache_dir="/model_cache")
    max_length = tokenizer.model_max_length
    
    # Load the model from the persistent cache with max_length
    zeroshot_classifier = pipeline("zero-shot-classification", 
                                model="MoritzLaurer/bge-m3-zeroshot-v2.0", 
                                # model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
                                # model="DAMO-NLP-SG/zero-shot-classify-SSTuning-base", 
                                cache_dir="/model_cache",
                                max_length=max_length)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    yield
    logger.info("Shutting down and cleaning up...")
    del zeroshot_classifier

app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    text: str
    category: Literal["Politics", "Lifestyle", "Children", "Upbringing", "Geo-Familiarity", "All"]
    hypothesis_template: str = "Based on this info from their dating bio, this person is best categorized as {}"

def preprocess_text(text):
    logger.debug(f"Preprocessing text: {text[:50]}...")  # Log first 50 characters
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Replace all whitespace (including newlines) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    processed_text = text.strip()
    logger.debug(f"Preprocessed text: {processed_text[:50]}...")  # Log first 50 characters
    return processed_text

@app.get('/')
async def welcome():
    logger.info("Welcome endpoint accessed")
    return "Welcome to our Dating Bio Classification API"

@app.post('/analyze')
async def classify_text(text_input: TextInput):
    logger.info(f"Received classification request for category: {text_input.category}")
    start_time = time.time()
    
    try:
        text_input_dict = jsonable_encoder(text_input)
        text_data = TextInput(**text_input_dict)

        if len(text_input.text) == 0:
            logger.warning("Received empty text input")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text_input.text)

        debug_info = {
            "preprocessed_text": preprocessed_text,
            "category_results": {}
        }

        if text_input.category == "All":
            logger.info("Processing all categories")
            results = {}
            for category, subcategories in CATEGORIES.items():
                logger.debug(f"Classifying for category: {category}")
                result = zeroshot_classifier(
                    preprocessed_text, 
                    subcategories, 
                    hypothesis_template=text_input.hypothesis_template
                )
                predicted_label = result['labels'][0]
                if predicted_label not in subcategories:
                    predicted_label = subcategories[-1]
                results[category] = predicted_label
                
                # Add debug information
                debug_info["category_results"][category] = {
                    "subcategories": result['labels'],
                    "scores": result['scores']
                }
            
            logger.info(f"Classification completed in {time.time() - start_time:.2f} seconds")
            return {
                "predicted_subcategories": results,
                "debug_info": debug_info
            }
        else:
            logger.info(f"Processing single category: {text_input.category}")
            subcategories = CATEGORIES[text_input.category]
            result = zeroshot_classifier(
                preprocessed_text, 
                subcategories, 
                hypothesis_template=text_input.hypothesis_template
            )
            predicted_label = result['labels'][0]
            if predicted_label not in subcategories:
                predicted_label = subcategories[-1]
            
            # Add debug information
            debug_info["category_results"][text_input.category] = {
                "subcategories": result['labels'],
                "scores": result['scores']
            }
            
            logger.info(f"Classification completed in {time.time() - start_time:.2f} seconds")
            return {
                "predicted_subcategory": predicted_label,
                "debug_info": debug_info
            }
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))