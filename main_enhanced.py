"""
Enhanced PolicyQuery AI Application
Uses modular architecture with utility classes and configuration management
"""

import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from pydantic import BaseModel, validator
import requests
import PyPDF2
import io

from config import Config, get_config
from utils import (
    DocumentProcessor, 
    EmbeddingManager, 
    VectorStoreManager, 
    LLMManager, 
    PerformanceMonitor,
    validate_url,
    sanitize_text
)

# Initialize configuration
config = get_config(os.getenv("ENVIRONMENT", "development"))

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    debug=config.DEBUG
)

# Security
security = HTTPBearer()

# Pydantic models with validation
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
    
    @validator('documents')
    def validate_documents(cls, v):
        if not validate_url(v):
            raise ValueError('Invalid document URL')
        if len(v) > config.VALIDATION['max_document_url_length']:
            raise ValueError('Document URL too long')
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > config.VALIDATION['max_questions']:
            raise ValueError(f'Too many questions. Maximum is {config.VALIDATION["max_questions"]}')
        
        for question in v:
            if len(question) < config.VALIDATION['min_question_length']:
                raise ValueError(f'Question too short. Minimum length is {config.VALIDATION["min_question_length"]}')
            if len(question) > config.VALIDATION['max_question_length']:
                raise ValueError(f'Question too long. Maximum length is {config.VALIDATION["max_question_length"]}')
        
        return v

class QueryResponse(BaseModel):
    answers: List[str]
    performance_metrics: dict = None

# Initialize managers
document_processor = DocumentProcessor(
    chunk_size=config.CHUNK_SIZE,
    overlap=config.CHUNK_OVERLAP
)

embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL)
vector_store_manager = VectorStoreManager(config.PINECONE_INDEX_NAME)
llm_manager = LLMManager(config.LLM_MODEL)
performance_monitor = PerformanceMonitor()

def verify_token(authorization: str = Header(...)):
    """Verify the authorization token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail=config.ERROR_MESSAGES["invalid_token"]
        )
    
    token = authorization.replace("Bearer ", "")
    if token != config.AUTH_TOKEN:
        raise HTTPException(
            status_code=401, 
            detail=config.ERROR_MESSAGES["invalid_token"]
        )
    
    return token

def download_and_parse_pdf(pdf_url: str) -> str:
    """Download and parse PDF content from URL"""
    try:
        logger.info(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        if len(response.content) > config.MAX_DOCUMENT_SIZE:
            raise ValueError("Document too large")
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_content += f"Page {page_num + 1}: {page_text}\n"
        
        logger.info(f"Successfully parsed PDF with {len(text_content)} characters")
        return sanitize_text(text_content)
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"{config.ERROR_MESSAGES['pdf_processing_error']}: {str(e)}"
        )

def process_document_queries(request: QueryRequest) -> QueryResponse:
    """Process document queries using the complete workflow"""
    
    # Step 1: Download and parse PDF
    performance_monitor.start_timer("pdf_processing")
    document_text = download_and_parse_pdf(request.documents)
    performance_monitor.end_timer("pdf_processing")
    
    # Step 2: Chunk the document
    performance_monitor.start_timer("document_chunking")
    chunks = document_processor.create_chunks(document_text)
    performance_monitor.end_timer("document_chunking")
    
    # Step 3: Generate embeddings
    performance_monitor.start_timer("embedding_generation")
    embeddings = embedding_manager.generate_embeddings([chunk.text for chunk in chunks])
    performance_monitor.end_timer("embedding_generation")
    
    # Step 4: Store embeddings in vector database
    performance_monitor.start_timer("vector_storage")
    vector_store_manager.store_embeddings(chunks, embeddings)
    performance_monitor.end_timer("vector_storage")
    
    # Step 5: Process each question
    answers = []
    for i, question in enumerate(request.questions):
        logger.info(f"Processing question {i + 1}/{len(request.questions)}: {question[:50]}...")
        
        # Find relevant clauses
        performance_monitor.start_timer(f"search_question_{i}")
        query_embedding = embedding_manager.generate_single_embedding(question)
        search_results = vector_store_manager.search_similar(
            query_embedding, 
            top_k=config.VECTOR_SEARCH_TOP_K
        )
        performance_monitor.end_timer(f"search_question_{i}")
        
        # Extract relevant clauses
        relevant_clauses = [match.metadata["text"] for match in search_results]
        
        # Generate answer using LLM
        performance_monitor.start_timer(f"llm_generation_{i}")
        answer = llm_manager.generate_answer(question, relevant_clauses)
        performance_monitor.end_timer(f"llm_generation_{i}")
        
        answers.append(answer)
        logger.info(f"Generated answer for question {i + 1}")
    
    # Prepare performance metrics
    metrics = {}
    for operation in performance_monitor.metrics:
        if "duration" in performance_monitor.metrics[operation]:
            metrics[operation] = performance_monitor.metrics[operation]["duration"]
    
    return QueryResponse(
        answers=answers,
        performance_metrics=metrics
    )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process document queries and return answers
    """
    try:
        logger.info(f"Processing query with {len(request.questions)} questions")
        
        # Validate configuration
        if not config.validate_config():
            raise HTTPException(
                status_code=500, 
                detail=config.ERROR_MESSAGES["missing_api_key"]
            )
        
        # Process the request
        result = process_document_queries(request)
        
        logger.info("Query processing completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=config.ERROR_MESSAGES["server_error"]
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": config.SUCCESS_MESSAGES["health_check"]}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "config_valid": config.validate_config()
    }
    
    # Check external services
    try:
        # Test embedding model
        test_embedding = embedding_manager.generate_single_embedding("test")
        health_status["embedding_model"] = "healthy"
    except Exception as e:
        health_status["embedding_model"] = f"error: {str(e)}"
    
    try:
        # Test LLM
        test_response = llm_manager.generate_answer("test", ["test context"])
        health_status["llm_model"] = "healthy"
    except Exception as e:
        health_status["llm_model"] = f"error: {str(e)}"
    
    return health_status

@app.get("/config")
async def get_config_info():
    """Get configuration information (for debugging)"""
    if not config.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "api_title": config.API_TITLE,
        "api_version": config.API_VERSION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "vector_search_top_k": config.VECTOR_SEARCH_TOP_K,
        "validation_rules": config.VALIDATION
    }

if __name__ == "__main__":
    import uvicorn
    
    # Validate configuration before starting
    if not config.validate_config():
        print("❌ Configuration validation failed. Please check your environment variables.")
        exit(1)
    
    print(f"🚀 Starting {config.API_TITLE} v{config.API_VERSION}")
    print(f"📍 Server: {config.HOST}:{config.PORT}")
    print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    uvicorn.run(
        "main_enhanced:app", 
        host=config.HOST, 
        port=config.PORT,
        reload=config.DEBUG
    ) 