from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = 'gh-faq-index'
DIMENSION = 768
NAMESPACE = "ns1"
MODEL_NAME = 'distilbert-base-uncased'
PINECONE_ENVIRONMENT = "us-east1-gcp"  # adjust as needed
PINECONE_INDEX_NAME = "gh-faq-index"

# Pinecone configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY) 