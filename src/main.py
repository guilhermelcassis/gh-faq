import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModel,
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from typing import List, Dict
import logging
from config.config import (
    PINECONE_API_KEY,
    MODEL_NAME,
    PINECONE_INDEX_NAME,
    NAMESPACE,
    DIMENSION,
    PINECONE_ENVIRONMENT
)
from src.rag_utils import RAGEnhancer
from datetime import datetime
import tensorflow as tf
import torch
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()

# Now log the API key (first few characters only for security)
api_key = os.getenv('PINECONE_API_KEY', '')
if api_key:
    logger.info(f"Loaded API key from .env: {api_key[:5]}...")
else:
    logger.error("No API key found in environment variables!")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure TensorFlow settings before any model loading
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")

# Initialize embedding model for semantic search
embed_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embed_model = AutoModel.from_pretrained(MODEL_NAME)

# Initialize T5 only for answer generation
qa_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
qa_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')

# Replace the Pinecone import with this conditional import
try:
    # Try new SDK first
    from pinecone import Pinecone
    use_new_sdk = True
except ImportError:
    # Fall back to old SDK
    import pinecone
    use_new_sdk = False

# Then modify your initialization code
if use_new_sdk:
    # New SDK initialization
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists and create it if it doesn't
    try:
        # List available indexes
        indexes = pc.list_indexes().names()
        logger.info(f"Available Pinecone indexes: {indexes}")
        
        if PINECONE_INDEX_NAME not in indexes:
            logger.info(f"Creating new index: {PINECONE_INDEX_NAME}")
            from pinecone import ServerlessSpec
            
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
        
        # Now connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error with Pinecone index: {str(e)}")
        raise
else:
    # Old SDK initialization
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # Check if index exists
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # Create index with old SDK
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric='cosine'
        )
        logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
    
    # Connect to index with old SDK
    index = pinecone.Index(PINECONE_INDEX_NAME)

# Initialize RAG enhancer
rag_enhancer = RAGEnhancer()

# Add this right after loading config values
logger.info(f"Using Pinecone index: {PINECONE_INDEX_NAME}")
logger.info(f"Using Pinecone environment: {PINECONE_ENVIRONMENT}")
logger.info(f"Using namespace: {NAMESPACE}")

# Add this line to use the correct namespace
if NAMESPACE == "":
    logger.info("Empty namespace detected, trying 'ns1' instead")
    NAMESPACE = "ns1"

class QuestionRequest(BaseModel):
    question: str

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings for semantic search."""
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state[0].mean(dim=0)
    
    return embeddings.tolist()

def generate_final_answer(contexts: List[Dict], question: str) -> str:
    """Generate a natural and factual answer strictly from provided context."""
    if not contexts or contexts[0]['score'] < 0.4:
        return "I'm sorry, but I couldn't find a sufficiently relevant answer to your question. Could you please rephrase it?"
    
    # Get the best matching context
    best_context = contexts[0]['metadata'].get('answer', '')
    category = contexts[0]['metadata'].get('category', '')
    
    # Special handling for common questions
    if 'airport' in question.lower() and 'Palermo' in best_context:
        return "The nearest airport is Aeroporto di Palermo (PMO). Bus transportation will be available from the airport on July 9th (check-in) and July 19th (check-out)."
    
    if any(word in question.lower() for word in ['transport', 'get there', 'how to get']):
        if 'bus transportation' in best_context.lower():
            return "Bus transportation will be available from Aeroporto di Palermo (PMO) on July 9th (check-in) and July 19th (check-out). This is the official transportation arrangement for getting to the school."
    
    if any(word in question.lower() for word in ['acomodation', 'accomodation', 'accommodation', 'stay']):
        if 'accommodation' in category.lower():
            return f"Here are the accommodation arrangements: {best_context}"
    
    if any(word in question.lower() for word in ['interview', 'selection', 'approved']):
        if 'Selection Process' in category:
            return f"Regarding the selection process: {best_context}"
    
    if any(word in question.lower() for word in ['do', 'activities', 'schedule']) and 'Daily Schedule' in category:
        return f"The daily activities at the school include: {best_context}"

    # Format context with better structure
    context_parts = []
    for ctx in contexts:
        metadata = ctx['metadata']
        if ctx['score'] >= 0.4:
            context_parts.append(metadata.get('answer', ''))
    
    # Join all relevant context parts
    full_context = ' '.join(context_parts)
    
    # Generate a natural response using the context
    prompt = (
        f"Question: {question}\n\n"
        f"Context: {full_context}\n\n"
        f"Instructions: Create a helpful and accurate response using ONLY the information provided above. "
        f"Do not add any information not present in the context."
    )

    # Generate answer with strict parameters
    inputs = qa_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = qa_model.generate(
        inputs.input_ids,
        max_length=256,
        num_beams=4,
        temperature=0.3,
        no_repeat_ngram_size=3,
        do_sample=False
    )
    
    return qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

def enhanced_search(question: str, index) -> List[Dict]:
    try:
        # Normalize and expand question
        normalized_question = rag_enhancer.normalize_question(question)
        question_lower = question.lower()
        
        # Generate embedding for vector search
        embedding = generate_embedding(normalized_question)
        
        # Get vector search results
        if use_new_sdk:
            results = index.query(
                vector=embedding,
                top_k=20,  # Get more results to filter
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = []
            for match in results.matches:
                matches.append({
                    'metadata': match.metadata,
                    'score': match.score
                })
        else:
            results = index.query(
                vector=embedding,
                top_k=20,  # Get more results to filter
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = [{'metadata': match['metadata'], 'score': match['score']} for match in results['matches']]
        
        # Log the raw results for debugging
        logger.info(f"Raw search results for '{question}': {matches[:2]}")  # Only log first 2 for brevity
        
        # Direct keyword matching for common question types
        if any(word in question_lower for word in ["cost", "price", "fee", "expensive", "cheap", "afford"]):
            logger.info("Detected fee related question")
            for match in matches:
                metadata = match['metadata']
                answer = metadata.get('answer', '').lower()
                category = metadata.get('category', '')
                
                if category == "Fees" or "850" in answer or "€" in answer or "euro" in answer or "fee" in answer:
                    logger.info(f"Found fee-related answer: {metadata.get('question', '')}")
                    match['score'] = 0.95
                    return [match]
        
        if any(word in question_lower for word in ["where", "location", "address", "place", "venue"]):
            logger.info("Detected location related question")
            for match in matches:
                metadata = match['metadata']
                category = metadata.get('category', '')
                
                if category == "Location and Travel":
                    logger.info(f"Found location-related answer: {metadata.get('question', '')}")
                    match['score'] = 0.95
                    return [match]
        
        if any(word in question_lower for word in ["schedule", "daily", "routine", "activities", "program", "what will i do"]):
            logger.info("Detected schedule related question")
            for match in matches:
                metadata = match['metadata']
                category = metadata.get('category', '')
                
                if category == "Daily Schedule":
                    logger.info(f"Found schedule-related answer: {metadata.get('question', '')}")
                    match['score'] = 0.95
                    return [match]
        
        if any(word in question_lower for word in ["accommodation", "stay", "housing", "dormitory", "sleep"]):
            logger.info("Detected accommodation related question")
            for match in matches:
                metadata = match['metadata']
                category = metadata.get('category', '')
                
                if category == "Accommodation":
                    logger.info(f"Found accommodation-related answer: {metadata.get('question', '')}")
                    match['score'] = 0.95
                    return [match]
        
        if any(word in question_lower for word in ["mission", "trip", "outreach", "after school"]):
            logger.info("Detected mission trip related question")
            for match in matches:
                metadata = match['metadata']
                category = metadata.get('category', '')
                
                if category == "Mission Trips":
                    logger.info(f"Found mission trip-related answer: {metadata.get('question', '')}")
                    match['score'] = 0.95
                    return [match]
                    
        # Try category-based matching using RAGEnhancer
        target_category = rag_enhancer.get_question_category(question)
        logger.info(f"Detected likely category for question: {target_category}")
        
        # If we have a target category, prioritize matches from that category
        if target_category and matches:
            category_matches = []
            for match in matches:
                metadata = match['metadata']
                if metadata.get('category', '') == target_category:
                    # Boost the score for category matches
                    match['score'] = 0.95  # High confidence for category matches
                    category_matches.append(match)
            
            if category_matches:
                logger.info(f"Found {len(category_matches)} matches in target category: {target_category}")
                return category_matches
        
        # Special handling for specific question types based on question_patterns.json
        for pattern_type, pattern_info in rag_enhancer.question_patterns.items():
            if any(p in question_lower for p in pattern_info['patterns']):
                # Look for answers containing the answer terms
                for match in matches:
                    metadata = match['metadata']
                    answer = metadata.get('answer', '').lower()
                    if any(term in answer for term in pattern_info['answer_terms']):
                        logger.info(f"Found {pattern_type}-related answer in content")
                        match['score'] = 0.95  # High confidence
                        return [match]
        
        # If no specific matches found, return the top match if it has a good score
        if matches and matches[0]['score'] > 0.8:
            logger.info(f"Using top match with score {matches[0]['score']}")
            return [matches[0]]
        
        # If we get here, try keyword-based fallback
        logger.warning(f"No good vector search results for '{question}', trying fallback...")
        fallback_results = keyword_fallback_search(question)
        if fallback_results:
            logger.info(f"Fallback search found {len(fallback_results)} results")
            return fallback_results
        
        # If all else fails, return the top match anyway
        if matches:
            logger.warning(f"No good matches found, returning top match as last resort")
            return [matches[0]]
        
        # If we get here, we have no matches at all
        logger.warning(f"No relevant matches found for '{question}'")
        return []
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {str(e)}", exc_info=True)
        return []

def log_search_metrics(question: str, results: List[Dict]):
    """Log search performance metrics."""
    try:
        metrics = {
            'question': question,
            'top_result_category': results[0]['metadata'].get('category') if results else None,
            'top_result_score': results[0]['score'] if results else 0,
            'results_count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Search metrics: {metrics}")
        
        # Could save metrics to database for analysis
        
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")

def format_answer(answer: str) -> str:
    """Format the answer to ensure special characters are displayed correctly."""
    # Replace common encoding issues
    answer = answer.replace('â', '€')
    
    # Ensure Euro symbol is properly encoded
    if '850' in answer and '€' not in answer and 'euro' not in answer.lower():
        answer = answer.replace('850', '850€')
    
    return answer

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """API endpoint to ask a question and get an answer."""
    try:
        question = request.question
        logger.info(f"Processing question: {question}")
        question_lower = question.lower()
        
        # Handle specific question types that need fallback answers
        if any(word in question_lower for word in ["age", "minimum age", "how old", "requirement"]):
            logger.info("Detected age requirement question")
            return get_fallback_answer(question)
            
        if any(word in question_lower for word in ["pack", "bring", "luggage", "clothes"]):
            logger.info("Detected packing question")
            return get_fallback_answer(question)
            
        if any(phrase in question_lower for phrase in ["what will happen", "what happens", "activities", "daily routine"]) and "school" in question_lower:
            logger.info("Detected school activities question")
            return get_fallback_answer(question)
            
        if any(phrase in question_lower for phrase in ["get to", "travel to", "directions", "transportation"]) and "school" in question_lower:
            logger.info("Detected transportation question")
            return get_fallback_answer(question)
        
        # Direct keyword matching for common question types
        if any(word in question_lower for word in ["cost", "price", "fee", "expensive", "cheap", "afford"]):
            logger.info("Direct match for fee question")
            fee_results = search_by_category("Fees")
            if fee_results:
                metadata = fee_results[0]['metadata']
                answer = metadata.get('answer', '')
                
                # Format the answer to fix encoding issues
                if 'â' in answer:
                    answer = answer.replace('â', '€')
                
                # Ensure Euro symbol is properly encoded
                if '850' in answer and '€' not in answer and 'euro' not in answer.lower():
                    answer = answer.replace('850', '850€')
                
                return {
                    "answer": answer,
                    "confidence": 0.95,
                    "sources": [{"category": "Fees", "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        if any(word in question_lower for word in ["where", "location", "address", "place", "venue"]):
            logger.info("Direct match for location question")
            location_results = search_by_category("Location and Travel")
            if location_results:
                metadata = location_results[0]['metadata']
                return {
                    "answer": metadata.get('answer', ''),
                    "confidence": 0.95,
                    "sources": [{"category": "Location and Travel", "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        if any(word in question_lower for word in ["schedule", "daily", "routine", "activities", "program", "what will i do"]):
            logger.info("Direct match for schedule question")
            schedule_results = search_by_category("Daily Schedule")
            if schedule_results:
                metadata = schedule_results[0]['metadata']
                return {
                    "answer": metadata.get('answer', ''),
                    "confidence": 0.95,
                    "sources": [{"category": "Daily Schedule", "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        if any(word in question_lower for word in ["accommodation", "stay", "housing", "dormitory", "sleep"]):
            logger.info("Direct match for accommodation question")
            accommodation_results = search_by_category("Accommodation")
            if accommodation_results:
                metadata = accommodation_results[0]['metadata']
                return {
                    "answer": metadata.get('answer', ''),
                    "confidence": 0.95,
                    "sources": [{"category": "Accommodation", "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        if any(word in question_lower for word in ["mission", "trip", "outreach", "after school"]):
            logger.info("Direct match for mission trip question")
            mission_results = search_by_category("Mission Trips")
            if mission_results:
                metadata = mission_results[0]['metadata']
                return {
                    "answer": metadata.get('answer', ''),
                    "confidence": 0.95,
                    "sources": [{"category": "Mission Trips", "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        # Try category-based matching using RAGEnhancer
        target_category = rag_enhancer.get_question_category(question)
        logger.info(f"Detected likely category: {target_category}")
        
        # Check for direct category matches
        if target_category != "General Information":
            logger.info(f"Direct match for {target_category} question")
            category_results = search_by_category(target_category)
            if category_results:
                metadata = category_results[0]['metadata']
                answer = metadata.get('answer', '')
                
                # Format the answer to fix encoding issues
                if 'â' in answer:
                    answer = answer.replace('â', '€')
                
                # Ensure Euro symbol is properly encoded
                if '850' in answer and '€' not in answer and 'euro' not in answer.lower():
                    answer = answer.replace('850', '850€')
                
                return {
                    "answer": answer,
                    "confidence": 0.95,
                    "sources": [{"category": target_category, "question": metadata.get('question', ''), "relevance": 0.95}]
                }
        
        # If no direct category match, use enhanced search
        search_results = enhanced_search(question, index)
        
        if not search_results:
            return {"answer": "I couldn't find any relevant information to answer your question."}
        
        # Get the top result
        top_result = search_results[0]
        metadata = top_result['metadata']
        
        # Log the question and top match
        logger.info(f"Question: {question}")
        logger.info(f"Top match: {metadata.get('category', 'Unknown')} - Score: {top_result['score']}")
        
        # Use the exact answer from the metadata
        answer = metadata.get('answer', '')
        
        # Format the answer to fix encoding issues
        if 'â' in answer:
            answer = answer.replace('â', '€')
        
        # Ensure Euro symbol is properly encoded
        if '850' in answer and '€' not in answer and 'euro' not in answer.lower():
            answer = answer.replace('850', '850€')
        
        # Create sources for transparency
        sources = []
        for i, result in enumerate(search_results[:2]):  # Include top 2 sources
            source_metadata = result['metadata']
            sources.append({
                "category": source_metadata.get('category', 'Unknown'),
                "question": source_metadata.get('question', ''),
                "relevance": result['score']
            })
        
        # Return the answer with confidence and sources
        return {
            "answer": answer,
            "confidence": top_result['score'],
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return {"answer": "Sorry, I encountered an error while processing your question."}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the system is working correctly."""
    return {"status": "healthy"}

@app.get("/index-stats")
async def get_index_stats():
    try:
        stats = index.describe_index_stats()
        # Extract only the serializable data from stats
        serializable_stats = {
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                ns: {
                    "vector_count": details.vector_count
                }
                for ns, details in stats.namespaces.items()
            }
        }
        return serializable_stats
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def ensure_faq_data_loaded():
    try:
        # Check if data exists in the index
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Check if we're running in Docker
        in_docker = os.path.exists('/.dockerenv')
        logger.info(f"Running in Docker: {in_docker}")
        
        # Force reload data if empty, few vectors, or in Docker
        if in_docker or not hasattr(stats, 'total_vector_count') or stats.total_vector_count < 5:
            logger.info("Loading FAQ data into Pinecone...")
            
            # Load FAQ data with better error handling
            import json
            import os
            
            # Try multiple possible locations for the FAQ file
            possible_paths = [
                'data/FAQ_structured.json',
                '/app/data/FAQ_structured.json',
                './data/FAQ_structured.json',
                '../data/FAQ_structured.json'
            ]
            
            faq_data = None
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        logger.info(f"Found FAQ data at: {path}")
                        with open(path, 'r') as f:
                            faq_data = json.load(f)
                        break
                except Exception as e:
                    logger.warning(f"Could not load from {path}: {str(e)}")
            
            if not faq_data:
                logger.error("Could not find FAQ data file!")
                return
            
            # Prepare vectors for upsert
            vectors = []
            for i, faq in enumerate(faq_data['faqs']):
                try:
                    # Generate embedding for the question
                    question_embedding = generate_embedding(faq['question'])
                    
                    # Create vector with metadata
                    vectors.append({
                        'id': f"faq_{i}",
                        'values': question_embedding,
                        'metadata': {
                            'question': faq['question'],
                            'question_variations': faq.get('question_variations', []),
                            'category': faq['category'],
                            'answer': faq['answer'],
                            'keywords': faq.get('keywords', [])
                        }
                    })
                    logger.info(f"Created embedding for: {faq['question']}")
                except Exception as e:
                    logger.error(f"Error creating embedding for FAQ {i}: {str(e)}")
            
            if not vectors:
                logger.error("No vectors created for FAQ data!")
                return
                
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
            
            # Upsert vectors to Pinecone
            try:
                if use_new_sdk:
                    index.upsert(vectors=vectors, namespace=NAMESPACE)
                else:
                    index.upsert(vectors=vectors, namespace=NAMESPACE)
                    
                logger.info(f"Successfully loaded {len(vectors)} FAQ entries into Pinecone")
                
                # Verify data was loaded
                stats = index.describe_index_stats()
                logger.info(f"Updated index stats: {stats}")
            except Exception as e:
                logger.error(f"Error upserting vectors to Pinecone: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error ensuring FAQ data: {str(e)}", exc_info=True)

# Call this function after index initialization
ensure_faq_data_loaded()

def load_question_patterns():
    """Load question patterns from JSON file with fallback to default patterns."""
    import json
    import os
    
    # Default patterns in case file loading fails
    default_patterns = {
        'cost': {
            'patterns': ['cost', 'price', 'fee', 'expensive', 'cheap', 'afford', 'payment', 'pay', 'money', 'euros', '€'],
            'categories': ['Fees'],
            'answer_terms': ['€', 'euro', 'cost', 'fee', 'price', '850']
        },
        'location': {
            'patterns': ['where', 'location', 'place', 'address', 'city', 'country', 'venue'],
            'categories': ['Location and Travel'],
            'answer_terms': ['located', 'location', 'address', 'italy', 'palermo']
        },
        'mission': {
            'patterns': ['mission', 'trip', 'travel', 'after school', 'outreach'],
            'categories': ['Mission Trips'],
            'answer_terms': ['mission', 'trip', 'optional', '10-day']
        },
        'interview': {
            'patterns': ['interview', 'selection', 'recruitment', 'process', 'apply', 'application'],
            'categories': ['Selection Process'],
            'answer_terms': ['process', 'selection', 'call', 'video', 'criteria']
        },
        'registration': {
            'patterns': ['register', 'sign up', 'join', 'apply', 'enroll', 'application'],
            'categories': ['Registration Process'],
            'answer_terms': ['sign up', 'register', 'apply', 'link', 'form']
        }
    }
    
    # Try to load from file
    possible_paths = [
        'data/question_patterns.json',
        '/app/data/question_patterns.json',
        './data/question_patterns.json',
        '../data/question_patterns.json'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Loading question patterns from: {path}")
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load patterns from {path}: {str(e)}")
    
    logger.warning("Using default question patterns as file could not be loaded")
    return default_patterns

# Load question patterns
question_patterns = load_question_patterns()

# Add this function to check Pinecone connectivity
def check_pinecone_connectivity():
    try:
        if use_new_sdk:
            indexes = pc.list_indexes().names()
            logger.info(f"Successfully connected to Pinecone. Available indexes: {indexes}")
            
            # Check if we're using the correct index name
            if PINECONE_INDEX_NAME not in indexes:
                logger.error(f"Index '{PINECONE_INDEX_NAME}' not found in available indexes: {indexes}")
                logger.error("Please check your PINECONE_INDEX_NAME environment variable")
                return False
                
            # Now connect to the index and check stats
            try:
                stats = index.describe_index_stats()
                logger.info(f"Index '{PINECONE_INDEX_NAME}' stats: {stats}")
                
                # Check if index is empty
                if hasattr(stats, 'total_vector_count') and stats.total_vector_count == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
                return True
            except Exception as e:
                logger.error(f"Error accessing index '{PINECONE_INDEX_NAME}': {str(e)}")
                return False
        else:
            # Old SDK
            indexes = pinecone.list_indexes()
            logger.info(f"Successfully connected to Pinecone. Available indexes: {indexes}")
            
            # Check if we're using the correct index name
            if PINECONE_INDEX_NAME not in indexes:
                logger.error(f"Index '{PINECONE_INDEX_NAME}' not found in available indexes: {indexes}")
                logger.error("Please check your PINECONE_INDEX_NAME environment variable")
                return False
                
            # Now check stats
            try:
                stats = index.describe_index_stats()
                logger.info(f"Index '{PINECONE_INDEX_NAME}' stats: {stats}")
                
                # Check if index is empty
                if 'total_vector_count' in stats and stats['total_vector_count'] == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
                return True
            except Exception as e:
                logger.error(f"Error accessing index '{PINECONE_INDEX_NAME}': {str(e)}")
                return False
    except Exception as e:
        logger.error(f"Error checking Pinecone connectivity: {str(e)}", exc_info=True)
        return False

# Call this function after initializing Pinecone
pinecone_connected = check_pinecone_connectivity()

def keyword_fallback_search(question: str) -> List[Dict]:
    """Fallback search using keywords when vector search fails."""
    try:
        # Load FAQ data
        import json
        import os
        
        # Try multiple possible locations for the FAQ file
        possible_paths = [
            'data/FAQ_structured.json',
            '/app/data/FAQ_structured.json',
            './data/FAQ_structured.json',
            '../data/FAQ_structured.json'
        ]
        
        faq_data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        faq_data = json.load(f)
                    break
            except Exception:
                continue
        
        if not faq_data:
            logger.error("Could not find FAQ data file for fallback search!")
            return []
        
        # Normalize question
        normalized_question = question.lower()
        
        # Direct keyword matching for common question types
        if any(word in normalized_question for word in ["cost", "price", "fee", "expensive", "cheap", "afford"]):
            logger.info("Fallback detected fee question")
            for faq in faq_data['faqs']:
                if faq['category'] == "Fees":
                    return [{
                        'metadata': {
                            'question': faq['question'],
                            'question_variations': faq.get('question_variations', []),
                            'category': faq['category'],
                            'answer': faq['answer'],
                            'keywords': faq.get('keywords', [])
                        },
                        'score': 0.95
                    }]
        
        if any(word in normalized_question for word in ["where", "location", "address", "place", "venue"]):
            logger.info("Fallback detected location question")
            for faq in faq_data['faqs']:
                if faq['category'] == "Location and Travel":
                    return [{
                        'metadata': {
                            'question': faq['question'],
                            'question_variations': faq.get('question_variations', []),
                            'category': faq['category'],
                            'answer': faq['answer'],
                            'keywords': faq.get('keywords', [])
                        },
                        'score': 0.95
                    }]
        
        if any(word in normalized_question for word in ["schedule", "daily", "routine", "activities", "program", "what will i do"]):
            logger.info("Fallback detected schedule question")
            for faq in faq_data['faqs']:
                if faq['category'] == "Daily Schedule":
                    return [{
                        'metadata': {
                            'question': faq['question'],
                            'question_variations': faq.get('question_variations', []),
                            'category': faq['category'],
                            'answer': faq['answer'],
                            'keywords': faq.get('keywords', [])
                        },
                        'score': 0.95
                    }]
        
        # Try category-based matching using RAGEnhancer
        target_category = rag_enhancer.get_question_category(question)
        logger.info(f"Fallback search detected category: {target_category}")
        
        # First try to find FAQs in the target category
        if target_category != "General Information":
            matching_faqs = []
            for faq in faq_data['faqs']:
                if faq['category'] == target_category:
                    matching_faqs.append({
                        'metadata': {
                            'question': faq['question'],
                            'question_variations': faq.get('question_variations', []),
                            'category': faq['category'],
                            'answer': faq['answer'],
                            'keywords': faq.get('keywords', [])
                        },
                        'score': 0.9
                    })
            
            if matching_faqs:
                logger.info(f"Fallback found {len(matching_faqs)} FAQs in category: {target_category}")
                return matching_faqs
        
        # Check for pattern matches from question_patterns.json
        for pattern_type, pattern_info in rag_enhancer.question_patterns.items():
            if any(p in normalized_question for p in pattern_info['patterns']):
                logger.info(f"Fallback detected {pattern_type} pattern in question: {question}")
                
                # Find FAQs in matching categories
                matching_faqs = []
                for faq in faq_data['faqs']:
                    if faq['category'] in pattern_info['categories']:
                        # Check if answer contains relevant terms
                        answer = faq['answer'].lower()
                        if any(term in answer for term in pattern_info['answer_terms']):
                            matching_faqs.append({
                                'metadata': {
                                    'question': faq['question'],
                                    'question_variations': faq.get('question_variations', []),
                                    'category': faq['category'],
                                    'answer': faq['answer'],
                                    'keywords': faq.get('keywords', [])
                                },
                                'score': 0.9
                            })
                
                if matching_faqs:
                    return matching_faqs
        
        # If no pattern match, try keyword matching
        keywords = normalized_question.split()
        scored_faqs = []
        
        for faq in faq_data['faqs']:
            # Check question and variations
            match_score = 0
            
            # Check direct question
            if any(kw in faq['question'].lower() for kw in keywords):
                match_score += 0.5
            
            # Check variations
            for var in faq.get('question_variations', []):
                if any(kw in var.lower() for kw in keywords):
                    match_score += 0.3
                    break
            
            # Check keywords
            for kw in faq.get('keywords', []):
                if kw.lower() in normalized_question:
                    match_score += 0.2
            
            if match_score > 0.4:
                scored_faqs.append({
                    'metadata': {
                        'question': faq['question'],
                        'question_variations': faq.get('question_variations', []),
                        'category': faq['category'],
                        'answer': faq['answer'],
                        'keywords': faq.get('keywords', [])
                    },
                    'score': match_score
                })
        
        # Sort and return top results
        scored_faqs.sort(key=lambda x: x['score'], reverse=True)
        return scored_faqs[:3]
        
    except Exception as e:
        logger.error(f"Error in fallback search: {str(e)}", exc_info=True)
        return []

# Add this function to force upload FAQ data to Pinecone
def force_upload_faq_to_pinecone():
    """Force upload FAQ data to Pinecone regardless of current state."""
    try:
        logger.info("Force uploading FAQ data to Pinecone...")
        
        # Load FAQ data
        import json
        import os
        
        # Try multiple possible locations for the FAQ file
        possible_paths = [
            'data/FAQ_structured.json',
            '/app/data/FAQ_structured.json',
            './data/FAQ_structured.json',
            '../data/FAQ_structured.json'
        ]
        
        faq_data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"Found FAQ data at: {path}")
                    with open(path, 'r') as f:
                        faq_data = json.load(f)
                    break
            except Exception as e:
                logger.warning(f"Could not load from {path}: {str(e)}")
        
        if not faq_data:
            logger.error("Could not find FAQ data file!")
            return False
        
        # Clear existing vectors if any
        try:
            if use_new_sdk:
                index.delete(delete_all=True, namespace=NAMESPACE)
            else:
                index.delete(delete_all=True, namespace=NAMESPACE)
            logger.info("Cleared existing vectors from index")
        except Exception as e:
            logger.warning(f"Error clearing vectors: {str(e)}")
        
        # Prepare vectors for upsert
        vectors = []
        for i, faq in enumerate(faq_data['faqs']):
            try:
                # Generate embedding for the question
                question_embedding = generate_embedding(faq['question'])
                
                # Create vector with metadata
                vectors.append({
                    'id': f"faq_{i}",
                    'values': question_embedding,
                    'metadata': {
                        'question': faq['question'],
                        'question_variations': faq.get('question_variations', []),
                        'category': faq['category'],
                        'answer': faq['answer'],
                        'keywords': faq.get('keywords', [])
                    }
                })
                logger.info(f"Created embedding for: {faq['question']}")
            except Exception as e:
                logger.error(f"Error creating embedding for FAQ {i}: {str(e)}")
        
        if not vectors:
            logger.error("No vectors created for FAQ data!")
            return False
            
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
        
        # Upsert vectors to Pinecone in batches to avoid timeouts
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                if use_new_sdk:
                    index.upsert(vectors=batch, namespace=NAMESPACE)
                else:
                    index.upsert(vectors=batch, namespace=NAMESPACE)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error upserting batch to Pinecone: {str(e)}")
                return False
                
        logger.info(f"Successfully loaded {len(vectors)} FAQ entries into Pinecone")
        
        # Verify data was loaded
        try:
            stats = index.describe_index_stats()
            logger.info(f"Updated index stats: {stats}")
            return True
        except Exception as e:
            logger.error(f"Error verifying data load: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error force uploading FAQ data: {str(e)}", exc_info=True)
        return False

# Call this function after initializing Pinecone
if not pinecone_connected:
    logger.warning("Pinecone connection failed, attempting force upload...")
    force_upload_faq_to_pinecone()

# Add this after initializing Pinecone
def diagnose_pinecone():
    """Diagnose Pinecone connection issues."""
    try:
        logger.info("Running Pinecone diagnostics...")
        
        # 1. Check API key
        if not PINECONE_API_KEY or len(PINECONE_API_KEY) < 10:
            logger.error(f"Invalid Pinecone API key: {PINECONE_API_KEY[:5]}...")
            return False
            
        # 2. Check if we can list indexes
        try:
            if use_new_sdk:
                indexes = pc.list_indexes().names()
            else:
                indexes = pinecone.list_indexes()
            logger.info(f"Successfully connected to Pinecone. Available indexes: {indexes}")
        except Exception as e:
            logger.error(f"Cannot list Pinecone indexes: {str(e)}")
            return False
            
        # 3. Check if our index exists
        if PINECONE_INDEX_NAME not in indexes:
            logger.error(f"Index '{PINECONE_INDEX_NAME}' not found in available indexes: {indexes}")
            return False
            
        # 4. Check index stats
        try:
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            # Check if index is empty
            if use_new_sdk:
                if not hasattr(stats, 'total_vector_count') or stats.total_vector_count == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
            else:
                if 'total_vector_count' not in stats or stats['total_vector_count'] == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
        except Exception as e:
            logger.error(f"Cannot get index stats: {str(e)}")
            return False
            
        # 5. Try a simple query
        try:
            # Create a test embedding
            test_embedding = [0.1] * DIMENSION
            
            # Query the index
            if use_new_sdk:
                results = index.query(
                    vector=test_embedding,
                    top_k=1,
                    include_metadata=True,
                    namespace=NAMESPACE
                )
                logger.info(f"Test query results: {results}")
            else:
                results = index.query(
                    vector=test_embedding,
                    top_k=1,
                    include_metadata=True,
                    namespace=NAMESPACE
                )
                logger.info(f"Test query results: {results}")
        except Exception as e:
            logger.error(f"Cannot query index: {str(e)}")
            return False
            
        logger.info("Pinecone diagnostics completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Pinecone diagnostics: {str(e)}", exc_info=True)
        return False

# Run diagnostics
pinecone_diagnostics = diagnose_pinecone()
if not pinecone_diagnostics:
    logger.warning("Pinecone diagnostics failed, will use fallback search")

def search_by_category(category: str) -> List[Dict]:
    """Search for answers in a specific category."""
    try:
        # Get all vectors from the index
        if use_new_sdk:
            stats = index.describe_index_stats()
            if not stats.namespaces.get(NAMESPACE):
                logger.warning(f"Namespace {NAMESPACE} not found in index")
                return []
        else:
            stats = index.describe_index_stats()
            if NAMESPACE not in stats.get('namespaces', {}):
                logger.warning(f"Namespace {NAMESPACE} not found in index")
                return []
        
        # Create a dummy embedding for the query
        dummy_embedding = [0.1] * DIMENSION
        
        # Query the index to get all vectors
        if use_new_sdk:
            results = index.query(
                vector=dummy_embedding,
                top_k=100,  # Get many results to filter
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = []
            for match in results.matches:
                matches.append({
                    'metadata': match.metadata,
                    'score': match.score
                })
        else:
            results = index.query(
                vector=dummy_embedding,
                top_k=100,  # Get many results to filter
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = [{'metadata': match['metadata'], 'score': match['score']} for match in results['matches']]
        
        # Filter by category
        category_matches = []
        for match in matches:
            metadata = match['metadata']
            if metadata.get('category', '') == category:
                match['score'] = 0.95  # High confidence for category matches
                category_matches.append(match)
        
        logger.info(f"Found {len(category_matches)} matches in category: {category}")
        return category_matches
        
    except Exception as e:
        logger.error(f"Error searching by category: {str(e)}", exc_info=True)
        return []

# Add this function to handle fallback responses for questions without direct answers
def get_fallback_answer(question: str) -> Dict:
    """Provide a fallback answer when no specific answer is found."""
    question_lower = question.lower()
    
    # Age requirement fallback
    if any(word in question_lower for word in ["age", "minimum age", "how old", "requirement"]):
        return {
            "answer": "There is no specific minimum age requirement mentioned for the Greenhouse school. The school is designed for people of various ages who want to experience revival and transformation. For specific eligibility requirements, please contact the organizers directly.",
            "confidence": 0.7,
            "sources": [{"category": "General Information", "question": "Age requirements", "relevance": 0.7}]
        }
    
    # Packing fallback
    if any(word in question_lower for word in ["pack", "bring", "luggage", "clothes"]):
        return {
            "answer": "Specific packing information will be provided to registered participants. Generally, you should bring comfortable clothes, personal items, and any materials specified in your acceptance information. The school is located in Palermo, Italy, so pack appropriate clothing for the local weather during your stay.",
            "confidence": 0.7,
            "sources": [{"category": "Practical Information", "question": "What to pack", "relevance": 0.7}]
        }
    
    # School activities fallback
    if any(phrase in question_lower for phrase in ["what will happen", "what happens", "activities", "daily routine"]):
        return {
            "answer": "The Greenhouse school includes teaching sessions, practical activation of spiritual gifts, worship, prayer, and community activities. The program is designed to provide both theoretical knowledge and practical experience in revival and transformation. Specific daily schedules will be provided to participants.",
            "confidence": 0.7,
            "sources": [{"category": "Daily Schedule", "question": "What happens at the school", "relevance": 0.7}]
        }
    
    # Transportation fallback
    if any(phrase in question_lower for phrase in ["get to", "travel to", "directions", "transportation"]):
        return {
            "answer": "The Greenhouse school is located in Palermo, Italy. Participants typically arrive via Palermo Airport (PMO). Detailed transportation information and directions will be provided to registered participants. If you need specific travel assistance, please contact the organizers directly.",
            "confidence": 0.7,
            "sources": [{"category": "Location and Travel", "question": "How to get to the school", "relevance": 0.7}]
        }
    
    # Default fallback
    return {
        "answer": "I don't have specific information to answer that question. Please contact the Greenhouse organizers directly for more details about your inquiry.",
        "confidence": 0.5,
        "sources": [{"category": "General Information", "question": "General inquiry", "relevance": 0.5}]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0", 
        port=8080, 
        workers=1,
        log_level="info"
    )
