import json
import logging
from embeddings import EmbeddingGenerator
from pinecone_ops import PineconeManager

"""
RAG_main.py

This script implements a FAQ search system that utilizes embeddings for better retrieval of questions and answers. 
It integrates with Pinecone for vector storage and retrieval.

Functions:
- load_faqs: Loads FAQ data from a JSON file.
- index_faqs: Indexes the FAQs into Pinecone after generating embeddings.
- clean_text: Cleans and splits text into question and answer.
- process_single_question: Processes a single user query and retrieves relevant FAQ matches.
- interactive_query: Manages user interaction for querying FAQs.
- reset_index: Resets the Pinecone index and reindexes the FAQs.
- main: The main entry point for the FAQ search system.

Usage:
Run the script to start the FAQ search system. Users can reset the index, search FAQs, or exit the program.
"""

logging.basicConfig(level=logging.INFO)

def load_faqs():
    # Load FAQ data
    with open('FAQ_structured.json') as f:
        return json.load(f)['faqs']

def index_faqs():
    faqs = load_faqs()
    
    # Initialize our classes
    embedding_generator = EmbeddingGenerator()
    pinecone_manager = PineconeManager()

    # Create index if it doesn't exist
    pinecone_manager.create_index()

    # Prepare vectors with embeddings
    vectors = []
    for i, faq in enumerate(faqs):
        if 'question' in faq and 'answer' in faq:
            # Let's improve how we structure the text for better retrieval
            text = f"Q: {faq['question']} A: {faq['answer']}"
            
            # Also store the category if available
            metadata = {
                'text': text,
                'question': faq['question'],
                'answer': faq['answer'],
                'category': faq.get('category', '')
            }
            
            embedding = embedding_generator.generate_embedding(text)
            
            vectors.append({
                "id": str(i),
                "values": embedding,
                "metadata": metadata
            })

    # Upsert to Pinecone
    if vectors:
        pinecone_manager.upsert_vectors(vectors)
        print(f"Indexed {len(vectors)} FAQ entries")
    else:
        logging.warning("No valid entries found in FAQs")

def clean_text(text):
    # Split into question and answer, handling various formats
    if ' A: ' in text:
        q, a = text.split(' A: ', 1)
        return q.replace('Q: ', '').strip(), a.strip()
    return text, ''  # Return full text if can't split

def process_single_question(query, embedding_generator, pinecone_manager):
    try:
        # Generate embedding for the query
        embedding = embedding_generator.generate_embedding(query)
        
        # Query Pinecone
        results = pinecone_manager.query(embedding, top_k=2)
        
        print(f"\nResults for: '{query}'")
        print("-" * 50)
        
        for i, match in enumerate(results['matches'], 1):
            score = match['score']
            metadata = match.get('metadata', {})
            
            print(f"\nMatch {i} (Similarity: {score:.3f})")
            
            # Safely access metadata fields with fallbacks
            if metadata.get('category'):
                print(f"Category: {metadata['category']}")
                
            # Handle both old and new metadata formats
            if 'question' in metadata and 'answer' in metadata:
                print(f"Question: {metadata['question']}")
                print(f"Answer: {metadata['answer']}")
            elif 'text' in metadata:
                # Handle old format where text contains both Q&A
                text = metadata['text']
                if 'Q:' in text and 'A:' in text:
                    parts = text.split('A:', 1)
                    question = parts[0].replace('Q:', '').strip()
                    answer = parts[1].strip()
                    print(f"Question: {question}")
                    print(f"Answer: {answer}")
                else:
                    print(f"Text: {text}")
            else:
                print("No text content available")
            
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        logging.error(f"Detailed error: {str(e)}", exc_info=True)

def interactive_query():
    embedding_generator = EmbeddingGenerator()
    pinecone_manager = PineconeManager()
    
    while True:
        # Get user input
        user_input = input("\nEnter your question(s) (or 'q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if not user_input:
            print("Please enter a question!")
            continue
        
        # Split multiple questions and clean them
        questions = [q.strip() + '?' for q in user_input.split('?') if q.strip()]
        
        if not questions:
            print("No valid questions found!")
            continue
        
        # Process each question separately
        for question in questions:
            process_single_question(question, embedding_generator, pinecone_manager)
            
        print("\n" + "=" * 70 + "\n")  # Separator between sets of results

def reset_index():
    try:
        pinecone_manager = PineconeManager()
        # Delete existing index if it exists
        try:
            pinecone_manager.delete_index()
            print("Existing index deleted.")
        except Exception as e:
            print(f"No existing index to delete or error: {e}")
        
        # Create fresh index
        pinecone_manager.create_index()
        print("New index created.")
        
        # Index the FAQs
        index_faqs()
    except Exception as e:
        print(f"Error resetting index: {e}")

def main():
    while True:
        print("\nFAQ Search System")
        print("1. Reset and Reindex FAQs")  # Changed from just Index
        print("2. Search FAQs")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            reset_index()
        elif choice == '2':
            interactive_query()
        elif choice == '3':
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 