"""
embedding_pinecone.py

This script generates embeddings for FAQ data using the DistilBERT model and stores them in a Pinecone index.
It loads FAQ data from a JSON file, processes the data to create embeddings, and upserts them into the specified Pinecone index.

Dependencies:
- pinecone
- python-dotenv
- transformers
- torch
- logging

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
4. Run the script to generate embeddings and store them in the Pinecone index.

Example:
    python embedding_pinecone.py
"""

from pinecone import Pinecone
import json
import os
from dotenv import load_dotenv
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import time
import logging  # Import logging for better error handling

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access the variables
api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Load FAQ data
with open('FAQ_structured.json') as f:
    faqs = json.load(f)['faqs']

# Prepare inputs for embeddings
inputs = [
    f"{d['question']} {d['answer']}"  # Combine question and answer into a single string
    for d in faqs if 'answer' in d and 'question' in d
]

# Check if inputs is empty
if not inputs:
    raise ValueError("No valid entries found in FAQs.")

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Tokenize inputs
encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')

# Generate embeddings
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Shape: (batch_size, hidden_size)

# Wait for the index to be ready
while not pc.describe_index('gh-faq-index').status['ready']:
    time.sleep(1)

# Prepare vectors with metadata
vectors = [
    {
        "id": str(i),  # Use a unique ID for each entry
        "values": embedding.tolist(),  # Use the embedding values
        "metadata": {'text': f"{d['question']} {d['answer']}"}  # Include relevant metadata
    }
    for i, (d, embedding) in enumerate(zip(faqs, embeddings))
]

# Get the index object
index = pc.Index('gh-faq-index')

# Upsert embeddings to Pinecone with a namespace
try:
    index.upsert(vectors=vectors, namespace="ns1")
    logging.info("Embeddings inserted successfully.")
except Exception as e:
    logging.error(f"Error inserting embeddings: {e}")