"""
Utility functions for PolicyQuery AI
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create overlapping chunks from text"""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            if chunk_text.strip():
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                chunk = DocumentChunk(
                    text=chunk_text,
                    start_index=i,
                    end_index=min(i + self.chunk_size, len(words)),
                    chunk_id=chunk_id,
                    metadata={
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "chunk_index": len(chunks)
                    }
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

class EmbeddingManager:
    """Manages embedding generation and storage"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise ValueError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]

class VectorStoreManager:
    """Manages vector database operations"""
    
    def __init__(self, index_name: str = "policy-query-ai"):
        self.index_name = index_name
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        try:
            import pinecone
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            raise
    
    def store_embeddings(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Store embeddings in vector database"""
        if not self.index:
            raise ValueError("Vector store not initialized")
        
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    **chunk.metadata
                }
            }
            vectors.append(vector)
        
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Stored {len(vectors)} embeddings in vector database")
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not self.index:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

class LLMManager:
    """Manages LLM operations"""
    
    def __init__(self, model_name: str = 'gemini-pro'):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM"""
        try:
            import google.generativeai as genai
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using LLM"""
        if not self.model:
            raise ValueError("LLM not initialized")
        
        context_text = "\n\n".join(context)
        
        prompt = f"""
        Based on the following document clauses, answer the question accurately and concisely.
        
        Document Clauses:
        {context_text}
        
        Question: {query}
        
        Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

class PerformanceMonitor:
    """Monitors system performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        import time
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        import time
        if operation in self.metrics:
            self.metrics[operation]["end"] = time.time()
            self.metrics[operation]["duration"] = (
                self.metrics[operation]["end"] - self.metrics[operation]["start"]
            )
    
    def get_duration(self, operation: str) -> Optional[float]:
        """Get duration of an operation"""
        if operation in self.metrics and "duration" in self.metrics[operation]:
            return self.metrics[operation]["duration"]
        return None
    
    def print_metrics(self):
        """Print all performance metrics"""
        print("\n📊 Performance Metrics:")
        for operation, data in self.metrics.items():
            if "duration" in data:
                print(f"  {operation}: {data['duration']:.2f}s")

def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def sanitize_text(text: str) -> str:
    """Sanitize text for safe processing"""
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    # Limit length
    if len(text) > 10000:
        text = text[:10000] + "..."
    return text.strip() 