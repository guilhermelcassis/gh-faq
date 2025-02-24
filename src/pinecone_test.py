"""
pinecone_test.py

This script tests the functionality of the Pinecone index by creating an index, loading FAQ data,
generating embeddings using a specified model, and upserting those embeddings into the Pinecone index.
It also retrieves and prints the embeddings to verify the operations.

Dependencies:
- pinecone
- python-dotenv
- json
- os
- transformers
- torch

Usage:
1. Ensure you have the required dependencies installed.
2. Create a .env file in the same directory with the following content:
   PINECONE_API_KEY=your_api_key_here
3. Prepare a JSON file named 'FAQ_structured.json' with the FAQ data structured as follows:
   {
       "faqs": [
           {"question": "Your question?", "answer": "Your answer."},
           ...
       ]
   }
4. Run the script to create the Pinecone index, generate embeddings, and verify the operations.

Example:
    python pinecone_test.py
"""

from pinecone import Pinecone, ServerlessSpec
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
api_key = os.getenv('PINECONE_API_KEY')
   
pc = Pinecone(api_key=api_key)

# Create a new index    
pc.create_index(
    name='gh-faq-index', # Replace with your index name
    dimension=768, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Load FAQ data
with open('FAQ_structured.json') as f:
    faqs = json.load(f)['faqs']
    
# Use both 'question' and 'answer' for embeddings
inputs = [
    {
        "text": f"{d['question']} {d['answer']}",  # Only include 'text'
    }
    for d in faqs if 'answer' in d and 'question' in d
]
    
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=inputs,
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])
print(len(embeddings[0]['values']))  # Should be 768
