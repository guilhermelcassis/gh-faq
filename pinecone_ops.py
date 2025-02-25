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

import os
import json
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'gh-faq-index')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
DIMENSION = int(os.getenv('DIMENSION', '768'))
MODEL_NAME = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
NAMESPACE = os.getenv('NAMESPACE', 'ns1')  # Use ns1 as default namespace

# Initialize embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Try new SDK first
try:
    from pinecone import Pinecone
    use_new_sdk = True
except ImportError:
    import pinecone
    use_new_sdk = False

def generate_embedding(text):
    """Generate embeddings for semantic search."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0].mean(dim=0)
    
    return embeddings.tolist()

def initialize_pinecone():
    """Initialize Pinecone and return the index."""
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
            return index
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
        return index

def upload_faq_data(index, faq_file_path='data/FAQ_structured.json'):
    """Upload FAQ data to Pinecone."""
    try:
        # Load FAQ data
        with open(faq_file_path, 'r') as f:
            faq_data = json.load(f)
        
        # Clear existing vectors if any
        try:
            if use_new_sdk:
                index.delete(delete_all=True, namespace=NAMESPACE)
            else:
                index.delete(delete_all=True, namespace=NAMESPACE)
            logger.info(f"Cleared existing vectors from namespace {NAMESPACE}")
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
            
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone namespace {NAMESPACE}...")
        
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
                
        logger.info(f"Successfully loaded {len(vectors)} FAQ entries into Pinecone namespace {NAMESPACE}")
        
        # Verify data was loaded
        try:
            stats = index.describe_index_stats()
            logger.info(f"Updated index stats: {stats}")
            return True
        except Exception as e:
            logger.error(f"Error verifying data load: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error uploading FAQ data: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # This script can be run directly to initialize and populate the index
    index = initialize_pinecone()
    upload_faq_data(index)