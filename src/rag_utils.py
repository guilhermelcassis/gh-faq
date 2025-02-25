import json
from pathlib import Path
from typing import List, Dict
from fuzzywuzzy import fuzz
import re
import logging
import numpy as np
from sentence_transformers import util
import os

logger = logging.getLogger(__name__)

class RAGEnhancer:
    def __init__(self):
        # Get the project root directory (2 levels up from this file)
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_dir = self.root_dir / "data"  # Update path to data directory
        self._load_configurations()

    def _load_configurations(self):
        """Load configuration files."""
        try:
            # Load category weights
            with open(self.config_dir / "category_weights.json", "r") as f:
                self.category_weights = json.load(f)

            # Load synonyms
            with open(self.config_dir / "synonyms.json", "r") as f:
                synonyms = json.load(f)
                self.transport_synonyms = synonyms["transport_synonyms"]
                self.activity_synonyms = synonyms["activity_synonyms"]
                self.accommodation_synonyms = synonyms.get("accommodation_synonyms", {})
                self.fee_synonyms = synonyms.get("fee_synonyms", {})
                self.selection_synonyms = synonyms.get("selection_synonyms", {})

            # Load category patterns
            with open(self.config_dir / "category_patterns.json", "r") as f:
                self.category_patterns = json.load(f)
                
            # Load question patterns
            with open(self.config_dir / "question_patterns.json", "r") as f:
                self.question_patterns = json.load(f)

            # Validate the loaded configurations
            self._validate_configurations()

        except Exception as e:
            print(f"Error loading configurations: {str(e)}")
            raise

    def _validate_configurations(self):
        """Validate the loaded configurations.

        This method checks for the presence of required keys in the loaded configurations
        and logs warnings for any missing keys. It ensures that the necessary data is
        available for the RAGEnhancer to function correctly.
        """
        try:
            # Load required keys from configuration
            with open(self.config_dir / "required_keys.json", "r") as f:
                required_keys = json.load(f)
            
            # Validate configurations
            for config_name, required in required_keys.items():
                if config_name == "category_weights":
                    config = self.category_weights
                elif config_name.endswith("_synonyms"):
                    synonym_type = config_name
                    config = getattr(self, synonym_type, {})
                elif config_name == "category_patterns":
                    config = self.category_patterns
                else:
                    config = {}
                    
                missing = [key for key in required if key not in config]
                if missing:
                    logger.warning(f"Missing required keys in {config_name}: {missing}")
                    
        except FileNotFoundError:
            logger.error("Required keys configuration file not found")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in required keys configuration")
            raise
        except Exception as e:
            logger.error(f"Error validating configurations: {e}")
            raise

    def find_semantic_clusters(self, questions: List[str], threshold: float = 0.8) -> List[List[str]]:
        """Group semantically similar questions.

        This method takes a list of questions and groups them into clusters based on
        semantic similarity, using a cosine similarity threshold.

        Args:
            questions (List[str]): A list of questions to cluster.
            threshold (float): The similarity threshold for clustering.

        Returns:
            List[List[str]]: A list of clusters, where each cluster is a list of similar questions.
        """
        embeddings = [self.generate_embedding(q) for q in questions]
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        clusters = []
        used = set()
        
        for i in range(len(questions)):
            if i in used:
                continue
                
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(questions)):
                if j not in used and similarity_matrix[i][j] >= threshold:
                    cluster.append(j)
                    used.add(j)
                    
            clusters.append([questions[idx] for idx in cluster])
        
        return clusters

    def fuzzy_keyword_match(self, question: str, keywords: List[str], threshold: int = 80) -> float:
        """Match keywords with fuzzy matching.

        This method compares a question against a list of keywords using fuzzy matching
        and returns the proportion of keywords that match above a specified threshold.

        Args:
            question (str): The question to match against keywords.
            keywords (List[str]): A list of keywords to match.
            threshold (int): The minimum score for a match to be considered valid.

        Returns:
            float: The proportion of matching keywords.
        """
        question_words = question.lower().split()
        max_scores = []
        
        for keyword in keywords:
            keyword_scores = [
                fuzz.ratio(keyword.lower(), word)
                for word in question_words
            ]
            max_scores.append(max(keyword_scores) if keyword_scores else 0)
        
        matching_keywords = sum(1 for score in max_scores if score >= threshold)
        return matching_keywords / len(keywords) if keywords else 0

    def create_overlapping_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Create overlapping chunks of text for better context.

        This method divides a given text into overlapping chunks to maintain context
        across segments.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The size of each chunk.
            overlap (int): The number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks

    def optimize_context_window(self, contexts: List[str], question: str, threshold: float = 0.3) -> str:
        """Optimize context window for better relevance.

        This method filters context chunks based on their relevance to a given question
        and concatenates the relevant chunks with special markers.

        Args:
            contexts (List[str]): A list of context chunks.
            question (str): The question to evaluate relevance against.
            threshold (float): The minimum relevance score for a chunk to be included.

        Returns:
            str: A concatenated string of relevant context chunks.
        """
        relevant_chunks = []
        
        for ctx in contexts:
            relevance_score = self.compute_relevance(ctx, question)
            if relevance_score > threshold:
                relevant_chunks.append(ctx)
        
        return self.concatenate_with_markers(relevant_chunks)

    def compute_relevance(self, context: str, question: str) -> float:
        """Compute simple relevance score using word overlap.

        This method calculates a relevance score based on the overlap of words
        between a context and a question.

        Args:
            context (str): The context to evaluate.
            question (str): The question to evaluate against the context.

        Returns:
            float: The relevance score between 0 and 1.
        """
        context_words = set(context.lower().split())
        question_words = set(question.lower().split())
        
        overlap = len(context_words.intersection(question_words))
        total = len(context_words.union(question_words))
        
        return overlap / total if total > 0 else 0

    def concatenate_with_markers(self, chunks: List[str]) -> str:
        """Concatenate chunks with special markers.

        This method joins a list of text chunks into a single string, separated by
        special markers for clarity.

        Args:
            chunks (List[str]): A list of text chunks to concatenate.

        Returns:
            str: A single string containing all chunks, separated by markers.
        """
        return "\n=====\n".join(chunks)

    def get_category_weight(self, question: str, category: str) -> float:
        """Get category-specific weight based on question.

        This method retrieves the weight associated with a specific category based
        on the presence of trigger words in the question.

        Args:
            question (str): The question to evaluate.
            category (str): The category for which to retrieve the weight.

        Returns:
            float: The weight for the specified category, defaulting to 1.0 if not found.
        """
        if category not in self.category_weights:
            return 1.0
            
        weights = self.category_weights[category]
        question_lower = question.lower()
        
        for trigger, weight in weights.items():
            if trigger in question_lower:
                return weight
                
        return 1.0

    def match_patterns(self, question: str, category: str) -> float:
        """Enhanced pattern matching with exact matches.

        This method checks a question against predefined patterns for a specific
        category, returning a score based on exact and regular matches.

        Args:
            question (str): The question to evaluate.
            category (str): The category containing patterns to match against.

        Returns:
            float: A score representing the match quality, with a base score for non-matching categories.
        """
        if category not in self.category_patterns:
            return 0.3  # Lower base score for non-matching categories
            
        patterns = self.category_patterns[category]
        question_lower = question.lower()
        
        # Check for exact matches first
        exact_matches = sum(1 for pattern in patterns 
                          if re.search(f"^{pattern}$", question_lower))
        if exact_matches > 0:
            return 1.0
            
        # Regular pattern matching
        matches = sum(1 for pattern in patterns 
                     if re.search(pattern, question_lower))
        
        base_score = matches / len(patterns) if patterns else 0.3
        return min(base_score * self.get_category_weight(question, category), 1.0)

    def compute_question_similarity(self, user_question: str, faq_question: str, variations: List[str]) -> float:
        """Compute similarity between user question and FAQ question + variations.

        This method evaluates the similarity between a user question and a FAQ question,
        including variations, to determine the best match.

        Args:
            user_question (str): The user's question.
            faq_question (str): The FAQ question to compare against.
            variations (List[str]): A list of variations of the FAQ question.

        Returns:
            float: The highest similarity score found.
        """
        # Clean and normalize questions
        user_q = self.normalize_text(user_question)
        faq_q = self.normalize_text(faq_question)
        
        # Calculate main question similarity
        main_similarity = self.compute_text_similarity(user_q, faq_q)
        
        # Calculate variation similarities
        variation_scores = [
            self.compute_text_similarity(user_q, self.normalize_text(var))
            for var in variations
        ] if variations else [0]
        
        # Return best match
        return max([main_similarity] + variation_scores)

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        This method prepares text for comparison by converting it to lowercase,
        removing punctuation, and normalizing whitespace.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.

        This method uses fuzzy string matching to calculate the similarity score
        between two pieces of text.

        Args:
            text1 (str): The first text to compare.
            text2 (str): The second text to compare.

        Returns:
            float: The similarity score between 0 and 1.
        """
        # Use fuzzy string matching
        return fuzz.ratio(text1, text2) / 100.0

    def normalize_question(self, question: str) -> str:
        """Normalize question and expand with synonyms.

        This method prepares a question for processing by converting it to lowercase,
        stripping whitespace, and replacing terms with their corresponding synonyms.

        Args:
            question (str): The question to normalize.

        Returns:
            str: The normalized question with synonyms included.
        """
        question = question.lower().strip()
        
        # Replace transport-related terms
        for key, synonyms in self.transport_synonyms.items():
            if key in question:
                for syn in synonyms:
                    question = f"{question} {syn}"
        
        # Replace activity-related terms
        for key, synonyms in self.activity_synonyms.items():
            if key in question:
                for syn in synonyms:
                    question = f"{question} {syn}"
                    
        # Replace accommodation-related terms
        for key, synonyms in self.accommodation_synonyms.items():
            if key in question:
                for syn in synonyms:
                    question = f"{question} {syn}"
                    
        # Replace fee-related terms
        for key, synonyms in self.fee_synonyms.items():
            if key in question:
                for syn in synonyms:
                    question = f"{question} {syn}"
                    
        # Replace selection-related terms
        for key, synonyms in self.selection_synonyms.items():
            if key in question:
                for syn in synonyms:
                    question = f"{question} {syn}"
        
        return question
        
    def get_question_category(self, question: str) -> str:
        """Determine the most likely category for a question.
        
        This method analyzes a question and returns the most likely category
        based on pattern matching from question_patterns.json.
        
        Args:
            question (str): The question to categorize.
            
        Returns:
            str: The most likely category for the question.
        """
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        for pattern_type, pattern_info in self.question_patterns.items():
            match_count = sum(1 for p in pattern_info['patterns'] if p in question_lower)
            score = match_count / len(pattern_info['patterns']) if pattern_info['patterns'] else 0
            
            if score > best_score:
                best_score = score
                best_match = pattern_type
                
        if best_match and best_score > 0.2:
            # Return the first category associated with this pattern type
            return self.question_patterns[best_match]['categories'][0]
        
        return "General Information"  # Default category 