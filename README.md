# Greenhouse FAQ Chatbot

This project provides a question-answering system for Greenhouse FAQ using vector search with Pinecone and natural language processing.

## Features

- Vector-based semantic search using Pinecone
- Fallback keyword search when vector search fails
- Natural language answer generation
- Docker support for easy deployment

## Setup

### Prerequisites

- Python 3.10+
- Pinecone API key
- HuggingFace API key (optional)

### Environment Variables

Create a `.env` file with the following variables: 

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=gh-faq-index
NAMESPACE=ns1
HUGGINGFACE_API_KEY=your_huggingface_api_key