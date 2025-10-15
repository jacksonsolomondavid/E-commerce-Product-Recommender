import json
import csv
from io import StringIO
from typing import List, Dict, Any, Optional, Type 

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session # NEW

# --- NEW: Database Imports (Assumes database.py is in the same folder) ---
from database import get_db, ProductDB, BehaviorDB, Base, engine 
# --------------------------------------------------------------------------


# --- Pydantic Models for Structured API Response ---

class LLMPromptPayload(BaseModel):
    """Defines the structure of the data sent to the Gemini API."""
    systemInstruction: str
    userQuery: str
    model: str = "gemini-2.5-flash-preview-05-20"

class RecommendationItem(BaseModel):
    """Defines a single recommended product and its associated data."""
    product_id: int
    name: str
    category: str
    price: float = Field(..., description="The price of the product.")
    user_behavior_context: Dict[str, List[str]]
    llm_prompt_payload: LLMPromptPayload = Field(
        ..., description="The structured prompt payload for the LLM."
    )

# --- FastAPI Application Setup ---

app = FastAPI(
    title="E-commerce Recommender API",
    description="Processes CSV data, runs heuristics, and prepares LLM prompts.",
    version="1.0.0"
)

# Configure CORS to allow communication from the HTML file running locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows communication from any origin (required for local HTML file access)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core Data Processing and Recommendation Logic (Modified for DB) ---

def parse_csv_to_dicts(csv_content: str, required_headers: List[str]) -> List[Dict[str, Any]]:
    """Reads a CSV string content and returns a list of dictionaries, enforcing headers."""
    # This is the helper that does the raw parsing, kept mostly as is.
    data = StringIO(csv_content)
        
    lines = data.readlines()
    if not lines:
        return []
        
    # Standardize headers: lowercase, trim whitespace, replace common delimiters with underscore
    header_line = lines[0].lower().replace(' ', '_').replace('-', '_')
    
    # Use StringIO again with the standardized header
    processed_content = StringIO(header_line + "".join(lines[1:]))
    reader = csv.DictReader(processed_content)

    # 1. Check if all required headers are present
    file_headers = [h.lower().strip().replace(' ', '_').replace('-', '_') for h in reader.fieldnames if h]
    available_headers = set(file_headers)
    
    for rh in required_headers:
        if rh not in available_headers:
            raise ValueError(f"Missing required CSV header: '{rh}'.")

    parsed_data = []
    for row in reader:
        processed_row = {}
        for header, value in row.items():
            if header:
                # Use the clean, standardized header name for the dictionary key
                clean_header = header.lower().strip().replace(' ', '_').replace('-', '_')
                processed_row[clean_header] = value.strip()
        
        # 2. Type conversion and validation
        if 'id' in processed_row:
            try: processed_row['id'] = int(processed_row['id'])
            except ValueError: processed_row['id'] = None 
        if 'price' in processed_row:
            try: processed_row['price'] = float(processed_row['price'])
            except ValueError: processed_row['price'] = 0.0 
            
        # Ensure 'user_id' is present and not empty for the log file context
        if 'user_id' in processed_row and not processed_row['user_id']:
             processed_row['user_id'] = None
            
        parsed_data.append(processed_row)
        
    # Filter out any rows missing critical data (like ID or user_id)
    return [row for row in parsed_data if row.get('id') is not None or (required_headers == ['user_id', 'action_type', 'value'] and row.get('user_id') is not None)]

def parse_csv_and_store(
    csv_content: str, 
    required_headers: List[str], 
    db_model: Type[Base], 
    db: Session
) -> List[Dict[str, Any]]:
    """Reads CSV content, stores it in the database, and returns the parsed data.
       The database table is cleared for this entity before new data is inserted."""
    
    # Use the existing parser to get the raw data dictionaries
    parsed_data = parse_csv_to_dicts(csv_content, required_headers) 
    
    # 1. Clear existing data for this type (catalog or behavior)
    # The database must be cleared to ensure only the most recent data is present
    db.query(db_model).delete()
    
    new_data_list = []
    
    for row in parsed_data:
        # Create DB model instances
        if db_model == ProductDB:
             # Map keys from standardized CSV row to DB model
             db_entry = ProductDB(
                 product_id=row['id'], 
                 name=row['name'], 
                 category=row['category'], 
                 price=row['price'],
                 # Assuming description might be an optional header
                 description=row.get('description', '') 
             )
        elif db_model == BehaviorDB:
             db_entry = BehaviorDB(
                 user_id=row['user_id'], 
                 action_type=row['action_type'], 
                 value=row['value']
             )
        else:
             continue 

        db.add(db_entry)
        new_data_list.append(row) 

    db.commit()
    return new_data_list 

def aggregate_user_behavior(user_id: str, db: Session) -> Optional[Dict[str, List[str]]]:
    """Aggregates user behavior data from the database."""
    
    # Query the database for the specific user ID
    log_data = db.query(BehaviorDB).filter(BehaviorDB.user_id == user_id).all()
    
    if not log_data:
        return None

    profile = {"view": set(), "search": set(), "purchase": set()}

    for entry in log_data:
        action = entry.action_type.lower()
        value = entry.value

        if value:
            # Check if action is valid and add value to the corresponding set
            if action in profile:
                profile[action].add(value)

    return {k: list(v) for k, v in profile.items()}

def get_recommendations_heuristic(catalog: List[Dict], user_behavior: Dict, limit: int = 3) -> List[Dict]:
    """Applies the scoring logic to the catalog based on user behavior. (NO CHANGE)"""
    
    primary_interests = user_behavior["view"] + [
        term for search in user_behavior["search"] for term in search.lower().split() if len(term) > 2
    ]
    
    scored_products = []
    for product in catalog:
        score = 0
        
        category = product.get("category", "")
        name = product.get("name", "").lower()
        description = product.get("description", "").lower()

        if category in user_behavior["view"]:
            score += 3
        
        search_terms = [t.lower() for t in user_behavior["search"]]
        product_text = name + description
        if any(term in product_text for term in search_terms if term):
            score += 2
        
        if any(interest in category or interest in description 
               for interest in primary_interests):
            score += 1
            
        if category in user_behavior["purchase"] and score > 0:
             score -= 1
        
        if score > 0:
            scored_products.append({
                **product,
                "score": score,
            })

    sorted_products = sorted(scored_products, key=lambda p: p["score"], reverse=True)
    return sorted_products[:limit]

def prepare_llm_prompt(product: Dict, user_behavior: Dict) -> LLMPromptPayload:
    """Constructs the structured prompt payload for the LLM. (NO CHANGE)"""
    
    system_prompt = (
        "You are an expert E-commerce Product Recommender System Analyst. "
        "Your task is to provide a compelling, concise, and professional single-paragraph "
        "explanation (max 50 words) to a customer, justifying why the following product "
        "was recommended to them based on their recent shopping behavior. Focus only on relevance and benefit."
    )

    user_query = (
        f"The user's recent shopping behavior includes: \n"
        f"Views: {', '.join(user_behavior['view'])}. \n"
        f"Searches: {', '.join(user_behavior['search'])}. \n"
        f"Purchases: {', '.join(user_behavior['purchase'])}. \n\n"
        f"The recommended product is: \n"
        f"Name: \"{product['name']}\" \n"
        f"Price: ${product.get('price', 0.0):.2f} \n"
        f"Category: {product['category']}. \n\n"
        f"Explain the recommendation."
    )

    return LLMPromptPayload(
        systemInstruction=system_prompt,
        userQuery=user_query
    )


# --- FastAPI Endpoint: Handles File Upload and Recommendation (MODIFIED) ---

@app.post(
    "/api/recommend/",
    response_model=List[RecommendationItem],
    summary="Uploads CSV data, stores it, and gets personalized recommendations"
)
async def get_recommendations_for_user(
    user_id: str = Form(..., description="The user ID to target for recommendations."),
    # Make files OPTIONAL
    catalog_file: Optional[UploadFile] = File(None, description="Optional: Product Catalog CSV file to upload."),
    behavior_file: Optional[UploadFile] = File(None, description="Optional: User Behavior Log CSV file to upload."),
    db: Session = Depends(get_db) # Inject the database session
):
    """
    Receives optional CSV files to store data, and returns recommendations 
    for the specified user ID, using either new data or existing DB data.
    """
    try:
        product_catalog = []
        
        # 1. Process and store/retrieve Product Catalog
        if catalog_file and catalog_file.size > 0:
            # SCENARIO 1: File Uploaded -> Store in DB
            catalog_content = (await catalog_file.read()).decode()
            product_catalog = parse_csv_and_store(
                catalog_content, ['id', 'name', 'price', 'category'], ProductDB, db
            )
        
        # SCENARIO 2: No File Uploaded -> Fetch from Database
        if not product_catalog:
             # Convert DB objects to dicts for heuristic logic
             product_catalog = [{
                 'id': p.product_id, 
                 'name': p.name, 
                 'price': p.price, 
                 'category': p.category, 
                 'description': p.description
             } for p in db.query(ProductDB).all()]
             
        if not product_catalog:
             raise HTTPException(status_code=404, detail="Product catalog is empty. Please upload a file or ensure data is in the database.")


        # 2. Process and store User Behavior Log (if uploaded)
        if behavior_file and behavior_file.size > 0:
            behavior_content = (await behavior_file.read()).decode()
            # Storing the new behavior log data
            parse_csv_and_store(
                behavior_content, ['user_id', 'action_type', 'value'], BehaviorDB, db
            )
        
        # 3. Aggregate user behavior (ALWAYS queries the database, using the newly stored data or old data)
        user_behavior = aggregate_user_behavior(user_id, db)
        
        if not user_behavior:
            raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found in the behavior log (database is empty or user has no activity).")

        # 4. Generate recommendations (Heuristic)
        recommended_products = get_recommendations_heuristic(product_catalog, user_behavior)
        
        # 5. Build final response structure (including LLM prompt payload)
        final_recommendations = []
        for product in recommended_products:
            llm_prompt = prepare_llm_prompt(product, user_behavior)
            
            final_recommendations.append(RecommendationItem(
                product_id=product["id"],
                name=product["name"],
                category=product["category"],
                price=product["price"],
                user_behavior_context=user_behavior,
                llm_prompt_payload=llm_prompt
            ))

        return final_recommendations
    
    except ValueError as ve:
        # Catch specific ValueErrors (like missing required headers)
        raise HTTPException(status_code=400, detail=f"CSV Parsing Error: {ve}")
    except HTTPException as e:
        # Re-raise explicit HTTP exceptions (400, 404)
        raise e
    except Exception as e:
        # Catch all other exceptions (internal logic errors)
        print(f"Internal server processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during recommendation processing.")