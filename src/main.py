import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
import torch
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
    DIMENSION
)
from rag_utils import RAGEnhancer
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize embedding model for semantic search
embed_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embed_model = AutoModel.from_pretrained(MODEL_NAME)

# Initialize T5 only for answer generation
qa_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
qa_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

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
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            namespace=NAMESPACE
        )
        
        # Enhanced scoring with better category matching
        scored_results = []
        for match in results.matches:
            metadata = match.metadata
            category = metadata.get('category', '')
            
            # Special handling for transportation questions
            if any(term in question.lower() for term in ['uber', 'taxi', 'transport']):
                if 'bus transportation' in metadata.get('answer', '').lower():
                    category_score = 1.0
                    category_weight = 2.0
            else:
                category_score = rag_enhancer.match_patterns(
                    normalized_question,
                    category
                )
                category_weight = rag_enhancer.get_category_weight(
                    normalized_question, 
                    category
                )
            
            # Calculate final score
            final_score = (
                match.score * 0.3 +
                category_score * 0.4 +
                rag_enhancer.compute_question_similarity(
                    normalized_question,
                    metadata.get('question', ''),
                    metadata.get('question_variations', [])
                ) * 0.3
            ) * category_weight
            
            scored_results.append({
                'metadata': metadata,
                'score': final_score
            })
        
        # Sort and filter
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return [r for r in scored_results if r['score'] > 0.4]
        
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

if __name__ == "____":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
