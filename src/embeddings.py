"""
embeddings.py

This module defines the EmbeddingGenerator class, which generates embeddings for input text using the DistilBERT model.
It provides methods to tokenize the input text, generate embeddings, and validate the generated embeddings.

Dependencies:
- torch
- transformers
- numpy
- config (for MODEL_NAME and DIMENSION)

Usage:
1. Ensure you have the required dependencies installed.
2. Import the EmbeddingGenerator class from this module.
3. Create an instance of the class.
4. Call the generate_embedding method with the input text to obtain the embeddings.

Example:
    from embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding("Your input text here.")
"""

import torch
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
from config.config import MODEL_NAME, DIMENSION

class EmbeddingGenerator:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.model = DistilBertModel.from_pretrained(MODEL_NAME)
        self.model.eval()  # Set model to evaluation mode
        
    def generate_embedding(self, text, debug=False):
        if not text:
            raise ValueError("Input text is empty.")
        
        # Tokenize the text
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            attention_mask = encoded_input['attention_mask']
            last_hidden_state = outputs.last_hidden_state
            
            if debug:
                print(f"Last hidden state shape: {last_hidden_state.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embedding = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.numpy().flatten()
            
            # Validate embedding
            self._validate_embedding(embedding)
            
            return embedding.tolist()
    
    def _validate_embedding(self, embedding):
        if embedding.shape[0] != DIMENSION:
            raise ValueError(f"Generated embedding has {embedding.shape[0]} dimensions instead of the required {DIMENSION}")
        
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            raise ValueError("Embedding contains NaN or infinite values") 