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
        
        # Generate embedding for semantic search
        embedding = generate_embedding(normalized_question)
        
        # Get initial results
        if use_new_sdk:
            # New SDK query syntax
            results = index.query(
                vector=embedding,
                top_k=15,
                include_metadata=True,
                namespace=NAMESPACE
            )
            # Convert to a consistent format
            matches = []
            for match in results.matches:
                matches.append({
                    'metadata': match.metadata,
                    'score': match.score
                })
        else:
            # Old SDK query syntax
            results = index.query(
                vector=embedding,
                top_k=15,
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = [{'metadata': match['metadata'], 'score': match['score']} for match in results['matches']]
        
        # Log the raw results for debugging
        logger.info(f"Raw search results for '{question}': {matches}")
        
        # If no results from vector search, try keyword-based fallback
        if not matches:
            logger.warning(f"No vector search results for '{question}', trying fallback...")
            fallback_results = keyword_fallback_search(question)
            if fallback_results:
                logger.info(f"Fallback search found {len(fallback_results)} results")
                return fallback_results
        
        # Use the loaded question patterns
        # Check for pattern matches in the question
        for pattern_type, pattern_info in question_patterns.items():
            if any(p in normalized_question.lower() for p in pattern_info['patterns']):
                logger.info(f"Detected {pattern_type} pattern in question: {question}")
                
                # First try to find a direct category match
                for match in matches:
                    metadata = match['metadata']
                    if metadata.get('category', '') in pattern_info['categories']:
                        # Check if answer contains relevant terms
                        answer = metadata.get('answer', '').lower()
                        if any(term in answer for term in pattern_info['answer_terms']):
                            logger.info(f"Found {pattern_type}-related answer with matching category")
                            return [{'metadata': metadata, 'score': 0.98}]
                
                # If no direct category match, look for answer terms in any result
                for match in matches:
                    metadata = match['metadata']
                    answer = metadata.get('answer', '').lower()
                    if any(term in answer for term in pattern_info['answer_terms']):
                        logger.info(f"Found {pattern_type}-related answer based on answer content")
                        return [{'metadata': metadata, 'score': 0.95}]
        
        # Enhanced scoring with keyword matching and question similarity
        scored_results = []
        
        # Extract key terms from the question
        question_terms = set(normalized_question.lower().split())
        
        for match in matches:
            metadata = match['metadata']
            category = metadata.get('category', '')
            
            # 1. Direct match check for exact questions or variations
            if question.lower() in [q.lower() for q in metadata.get('question_variations', [])]:
                logger.info(f"Found direct match for question: {question}")
                return [{'metadata': metadata, 'score': 1.0}]
            
            # Check for partial matches in question variations
            variation_scores = []
            for variation in metadata.get('question_variations', []):
                variation_terms = set(variation.lower().split())
                overlap = len(question_terms.intersection(variation_terms)) / max(1, len(question_terms))
                variation_scores.append(overlap)
            
            variation_score = max(variation_scores) if variation_scores else 0
            
            # 2. Check for keyword matches
            keyword_score = 0
            keywords = metadata.get('keywords', [])
            if keywords:
                matching_keywords = sum(1 for k in keywords if k.lower() in normalized_question.lower())
                if matching_keywords > 0:
                    keyword_score = matching_keywords / len(keywords)
            
            # 3. Check for question term overlap
            question_text = metadata.get('question', '').lower()
            question_terms_in_metadata = set(question_text.split())
            term_overlap = len(question_terms.intersection(question_terms_in_metadata)) / max(1, len(question_terms))
            
            # 4. Check for category relevance
            category_terms = set(category.lower().split())
            category_match = len(question_terms.intersection(category_terms)) / max(1, len(category_terms))
            
            # 5. Check for answer relevance to question
            answer_text = metadata.get('answer', '').lower()
            
            # Calculate final score with weighted components
            final_score = (
                match['score'] * 0.3 +         # Vector similarity
                keyword_score * 0.15 +         # Keyword matching
                term_overlap * 0.15 +          # Question term overlap
                category_match * 0.15 +        # Category relevance
                variation_score * 0.25         # Question variation match
            )
            
            scored_results.append({
                'metadata': metadata,
                'score': final_score
            })
        
        # Sort and filter
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        filtered_results = [r for r in scored_results if r['score'] > 0.5]  # Increase threshold
        
        logger.info(f"Final scored results: {filtered_results}")
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {str(e)}", exc_info=True)
        raise

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

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        logger.info(f"Processing question: {question}")
        
        # Get search results
        search_results = enhanced_search(question, index)
        
        if not search_results:
            return {"answer": "I couldn't find any relevant information to answer your question."}
        
        # Use top result for simple questions, multiple results for complex ones
        best_matches = search_results[:2]
        final_answer = generate_final_answer(best_matches, question)
        
        # Log for debugging
        logger.info(f"Question: {question}")
        logger.info(f"Top match: {best_matches[0]['metadata'].get('category')} - Score: {best_matches[0]['score']}")
        
        return {
            "answer": final_answer,
            "confidence": best_matches[0]['score'],
            "sources": [
                {
                    "category": match['metadata'].get('category'),
                    "question": match['metadata'].get('question'),
                    "relevance": match['score']
                }
                for match in best_matches
            ]
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

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
            
            if PINECONE_INDEX_NAME in indexes:
                stats = index.describe_index_stats()
                logger.info(f"Index '{PINECONE_INDEX_NAME}' stats: {stats}")
                
                # Check if index is empty
                if hasattr(stats, 'total_vector_count') and stats.total_vector_count == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
                return True
            else:
                logger.error(f"Index '{PINECONE_INDEX_NAME}' not found in available indexes!")
                return False
        else:
            # Old SDK
            indexes = pinecone.list_indexes()
            logger.info(f"Successfully connected to Pinecone. Available indexes: {indexes}")
            
            if PINECONE_INDEX_NAME in indexes:
                stats = index.describe_index_stats()
                logger.info(f"Index '{PINECONE_INDEX_NAME}' stats: {stats}")
                
                # Check if index is empty
                if 'total_vector_count' in stats and stats['total_vector_count'] == 0:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' exists but is empty!")
                    return False
                return True
            else:
                logger.error(f"Index '{PINECONE_INDEX_NAME}' not found in available indexes!")
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
        
        # Check for pattern matches
        for pattern_type, pattern_info in question_patterns.items():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0", 
        port=8000, 
        workers=1,
        log_level="info"
    )
