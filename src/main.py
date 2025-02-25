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
                top_k=5,
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
                top_k=5,
                include_metadata=True,
                namespace=NAMESPACE
            )
            matches = [{'metadata': match['metadata'], 'score': match['score']} for match in results['matches']]
        
        # Log the raw results for debugging
        logger.info(f"Raw search results for '{question}': {matches}")
        
        # Enhanced scoring with better category matching
        scored_results = []
        for match in matches:
            metadata = match['metadata']
            category = metadata.get('category', '')
            
            # Direct match check for exact questions
            if question.lower() in [q.lower() for q in metadata.get('question_variations', [])]:
                logger.info(f"Found direct match for question: {question}")
                return [{'metadata': metadata, 'score': 1.0}]
            
            # Calculate final score
            final_score = match['score']
            
            scored_results.append({
                'metadata': metadata,
                'score': final_score
            })
        
        # Sort and filter
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        filtered_results = [r for r in scored_results if r['score'] > 0.4]
        
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

# Add this after your index initialization
def ensure_faq_data_loaded():
    try:
        # Check if data exists in the index
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            logger.info("Index is empty, loading FAQ data...")
            
            # Load FAQ data
            import json
            with open('data/FAQ_structured.json', 'r') as f:
                faq_data = json.load(f)
            
            # Prepare vectors for upsert
            vectors = []
            for i, faq in enumerate(faq_data['faqs']):
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
            
            # Upsert vectors to Pinecone
            if use_new_sdk:
                index.upsert(vectors=vectors, namespace=NAMESPACE)
            else:
                index.upsert(vectors=vectors, namespace=NAMESPACE)
                
            logger.info(f"Loaded {len(vectors)} FAQ entries into Pinecone")
    except Exception as e:
        logger.error(f"Error ensuring FAQ data: {str(e)}")

# Call this function after index initialization
ensure_faq_data_loaded()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0", 
        port=8000, 
        workers=1,
        log_level="info"
    )
