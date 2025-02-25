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


### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the index setup script: `python pinecone_ops.py`
4. Start the API server: `python -m src.main`

### Docker Deployment

1. Build the Docker image: `docker build -t faq-project .`
2. Run the container: `docker run -p 8000:8000 --env-file ./.env faq-project`

## API Endpoints

- `POST /ask`: Ask a question and get an answer
- `GET /health`: Check the health of the system

## Project Structure

- `src/main.py`: Main FastAPI application
- `src/rag_utils.py`: RAG enhancement utilities
- `pinecone_ops.py`: Pinecone operations for index management
- `config/config.py`: Configuration settings
- `data/`: Contains FAQ data and patterns