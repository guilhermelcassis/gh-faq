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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from pathlib import Path

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

def load_faq_to_pinecone(index_name: str):
    """Load FAQ data into Pinecone index."""
    logger.info("Connecting to Pinecone...")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    json_file = root_dir / "data" / "FAQ_structured.json"
    
    logger.info(f"Loading FAQ data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        # Initialize Pinecone manager
        manager = PineconeManager()
        
        # Process each FAQ entry
        vectors = []
        for i, faq in enumerate(faq_data['faqs']):
            # Create vector for main question
            question_vector = create_vector(
                faq['question'],
                f"q_{i}",
                {
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'category': faq.get('category', ''),
                    'type': 'question'
                },
                tokenizer,
                model
            )
            vectors.append(question_vector)
            
            # Create vectors for question variations
            for j, variation in enumerate(faq.get('question_variations', [])):
                var_vector = create_vector(
                    variation,
                    f"q_{i}_var_{j}",
                    {
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'category': faq.get('category', ''),
                        'type': 'variation'
                    },
                    tokenizer,
                    model
                )
                vectors.append(var_vector)
            
            # Create vector for answer
            answer_vector = create_vector(
                faq['answer'],
                f"a_{i}",
                {
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'category': faq.get('category', ''),
                    'type': 'answer'
                },
                tokenizer,
                model
            )
            vectors.append(answer_vector)
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            manager.upsert_vectors(batch)
            logger.info(f"Upserted batch {i//batch_size + 1}")
            
    except Exception as e:
        logger.error(f"Error in load_faq_to_pinecone: {str(e)}")
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
        load_faq_to_pinecone("data/FAQ_structured.json")
        logger.info("Successfully loaded FAQ data to Pinecone")
    except Exception as e:
        logger.error(f"Failed to load FAQ data: {str(e)}") 