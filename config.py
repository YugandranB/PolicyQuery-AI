"""
Configuration settings for PolicyQuery AI
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Configuration
    API_TITLE = "PolicyQuery AI"
    API_DESCRIPTION = "LLM-Powered Intelligent Query-Retrieval System"
    API_VERSION = "1.0.0"
    BASE_URL = "http://localhost:8000"
    API_PREFIX = "/api/v1"
    
    # Authentication
    AUTH_TOKEN = "3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc"
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Model Configuration
    LLM_MODEL = "gemini-pro"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Vector Database
    PINECONE_INDEX_NAME = "policy-query-ai"
    VECTOR_SEARCH_TOP_K = 3
    
    # Performance Settings
    REQUEST_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_REQUESTS = 10
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Error Messages
    ERROR_MESSAGES = {
        "invalid_token": "Invalid authentication token",
        "missing_api_key": "Missing required API key",
        "pdf_processing_error": "Error processing PDF document",
        "embedding_error": "Error generating embeddings",
        "llm_error": "Error generating answer",
        "vector_search_error": "Error searching vector database",
        "invalid_url": "Invalid document URL provided",
        "empty_questions": "No questions provided",
        "server_error": "Internal server error"
    }
    
    # Success Messages
    SUCCESS_MESSAGES = {
        "health_check": "PolicyQuery AI is running",
        "request_processed": "Request processed successfully"
    }
    
    # API Endpoints
    ENDPOINTS = {
        "health": "/health",
        "root": "/",
        "process_query": "/hackrx/run"
    }
    
    # Validation Rules
    VALIDATION = {
        "max_questions": 10,
        "min_question_length": 5,
        "max_question_length": 500,
        "max_document_url_length": 2000
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_keys = [
            "GEMINI_API_KEY",
            "PINECONE_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing required configuration: {missing_keys}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    @classmethod
    def get_api_url(cls, endpoint: str) -> str:
        """Get full API URL for an endpoint"""
        return f"{cls.BASE_URL}{cls.API_PREFIX}{endpoint}"
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get default headers for API requests"""
        return {
            "Authorization": f"Bearer {cls.AUTH_TOKEN}",
            "Content-Type": "application/json"
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    REQUEST_TIMEOUT = 600  # 10 minutes

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    PINECONE_INDEX_NAME = "policy-query-ai-test"

# Configuration mapping
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

def get_config(environment: str = "development") -> Config:
    """Get configuration for the specified environment"""
    return config_map.get(environment, DevelopmentConfig) 