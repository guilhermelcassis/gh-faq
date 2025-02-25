"""
create_pinecone_index.py

This script initializes a Pinecone index for storing and querying vector embeddings.
It checks if the specified index already exists and creates it if it does not.

Dependencies:
- pinecone
- python-dotenv
- logging

Usage:
1. Ensure you have the required dependencies installed.
2. Create a .env file in the same directory with the following content:
   PINECONE_API_KEY=your_api_key_here
3. Run the script to create the Pinecone index.

Example:
    python create_pinecone_index.py
"""

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import logging  # Import logging for better error handling

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access the variables
api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create a new index if it doesn't exist
index_name = 'gh-faq-index'

try:
    pc.create_index(
        name=index_name,
        dimension=768,  # Replace with your model dimensions
        metric="cosine",  # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    logging.info(f"Created index '{index_name}'.")
except Exception as e:
    if "already exists" in str(e):
        logging.info(f"Index '{index_name}' already exists.")
    else:
        logging.error(f"An error occurred while creating the index: {e}")