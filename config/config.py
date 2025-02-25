from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = 'gh-faq-index'
DIMENSION = int(os.getenv('DIMENSION', '768'))
NAMESPACE = os.getenv('NAMESPACE', 'ns1')
MODEL_NAME = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")