"""
pinecone_ops.py

This module provides the PineconeManager class for managing a Pinecone index, including operations such as
creating, deleting, and querying the index, as well as upserting vectors. It also includes a function to load
FAQ data from a JSON file and generate embeddings for storage in the Pinecone index.

Dependencies:
- pinecone
- transformers
- torch
- nltk
- logging
- json
- time

Usage:
1. Ensure you have the required dependencies installed.
2. Create an instance of the PineconeManager class to manage the Pinecone index.
3. Use the methods provided to create, delete, or query the index, or to upsert vectors.
4. Use the load_faq_to_pinecone function to load FAQ data from a JSON file into the index.

Example:
    from pinecone_ops import PineconeManager, load_faq_to_pinecone

    manager = PineconeManager()
    manager.create_index()
    load_faq_to_pinecone("FAQ_structured.json")
"""

import logging
from config.config import pc, INDEX_NAME, NAMESPACE, DIMENSION, PINECONE_API_KEY, PINECONE_INDEX_NAME, MODEL_NAME
from pinecone import ServerlessSpec
import time
import json
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
from pinecone import Pinecone
import re
from nltk.tokenize import word_tokenize
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self):
        self.index = pc.Index(INDEX_NAME)
    
    def delete_index(self):
        """Delete the existing index if it exists"""
        try:
            pc.delete_index(INDEX_NAME)
            logging.info(f"Deleted index '{INDEX_NAME}'")
        except Exception as e:
            logging.error(f"Error deleting index: {e}")
            raise

    def create_index(self):
        """Create a new index"""
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logging.info(f"Created index '{INDEX_NAME}'")
            # Wait for the index to be ready
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        except Exception as e:
            if "already exists" in str(e):
                logging.info(f"Index '{INDEX_NAME}' already exists")
            else:
                raise
    
    def upsert_vectors(self, vectors):
        try:
            self.index.upsert(vectors=vectors, namespace=NAMESPACE)
            logging.info(f"Successfully upserted {len(vectors)} vectors")
        except Exception as e:
            logging.error(f"Error upserting vectors: {e}")
            raise
    
    def query(self, vector, top_k=5):
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=NAMESPACE,
                include_metadata=True
            )
            return results
        except Exception as e:
            logging.error(f"Error querying index: {e}")
            raise 

def load_faq_to_pinecone(json_file: str):
    """Load FAQ data from JSON file to Pinecone index."""
    try:
        # Initialize models
        logger.info("Initializing models...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Load FAQ data
        logger.info(f"Loading FAQ data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
            
        # Convert to list if it's a dictionary
        if isinstance(faq_data, dict):
            faq_data = list(faq_data.values())
        
        # Process and upsert vectors
        batch_size = 100
        total_processed = 0
        vectors = []
        
        logger.info(f"Processing {len(faq_data)} FAQ items...")
        
        for idx, item in enumerate(faq_data['faqs']):
            # Create vectors for main question and variations
            vectors = []
            
            # Combine all relevant information
            base_text = f"""
            Category: {item['category']}
            Keywords: {', '.join(item['keywords'])}
            Question: {item['question']}
            Answer: {item['answer']}
            """
            
            # Create main vector
            main_vector = create_vector(
                text=base_text,
                id=f"faq_{idx}_main",
                metadata={
                    'question': item['question'],
                    'answer': item['answer'],
                    'category': item['category'],
                    'keywords': item['keywords'],
                    'is_main': True
                },
                tokenizer=tokenizer,
                model=model
            )
            vectors.append(main_vector)
            
            # Create vectors for variations
            for var_idx, variation in enumerate(item['question_variations']):
                var_text = f"""
                Category: {item['category']}
                Keywords: {', '.join(item['keywords'])}
                Question: {variation}
                Related Question: {item['question']}
                Answer: {item['answer']}
                """
                
                var_vector = create_vector(
                    text=var_text,
                    id=f"faq_{idx}_var_{var_idx}",
                    metadata={
                        'question': variation,
                        'main_question': item['question'],
                        'answer': item['answer'],
                        'category': item['category'],
                        'keywords': item['keywords'],
                        'is_variation': True
                    },
                    tokenizer=tokenizer,
                    model=model
                )
                vectors.append(var_vector)
            
            # Batch upsert
            index.upsert(vectors=vectors)
            logger.info(f"Uploaded vectors for FAQ {idx}: 1 main + {len(item['question_variations'])} variations")
        
        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats after upload: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in load_faq_to_pinecone: {str(e)}", exc_info=True)
        raise

def create_vector(text: str, id: str, metadata: dict, tokenizer, model) -> dict:
    """Create a vector with enhanced context."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[0].mean(dim=0)
    
    return {
        'id': id,
        'values': embedding.tolist(),
        'metadata': metadata
    }

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text."""
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    
    # Keep nouns and verbs
    important_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    keywords = [word for word, tag in tagged if tag in important_tags]
    
    return keywords

if __name__ == "__main__":
    try:
        load_faq_to_pinecone("FAQ_structured.json")
        logger.info("Successfully loaded FAQ data to Pinecone")
    except Exception as e:
        logger.error(f"Failed to load FAQ data: {str(e)}") 