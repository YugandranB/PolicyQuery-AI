import os
import json
import requests
import PyPDF2
import io
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import google.generativeai as genai
import pinecone
import numpy as np
from dotenv import load_dotenv
import hashlib
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PolicyQuery AI",
    description="LLM-Powered Intelligent Query-Retrieval System",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
AUTH_TOKEN = "3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc"

# Global variables for local storage
local_chunks = []
local_embeddings = []

# Initialize models
genai.configure(api_key="AIzaSyBZ9HLWnmFg-Dvks7iUrpGQFxBs-ogqlvc")
model = genai.GenerativeModel('gemini-1.5-flash')

# Simple embedding function using hash-based approach
def simple_embedding(text: str) -> List[float]:
    """Generate a simple embedding using hash-based approach"""
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to a list of floats (384 dimensions like all-MiniLM-L6-v2)
    embedding = []
    for i in range(0, len(text_hash), 2):
        if len(embedding) >= 384:
            break
        hex_val = text_hash[i:i+2]
        embedding.append(float(int(hex_val, 16)) / 255.0)
    
    # Pad or truncate to exactly 384 dimensions
    while len(embedding) < 384:
        embedding.append(0.0)
    
    return embedding[:384]

# Initialize Pinecone
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_name = "policy-query-ai"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=384,  # Dimension for our simple embedding
            metric="cosine"
        )
    index = pinecone.Index(index_name)
    PINECONE_AVAILABLE = True
    print("✅ Pinecone initialized successfully")
except Exception as e:
    print(f"⚠️  Pinecone initialization failed: {e}")
    print("⚠️  Running in local mode without vector database")
    PINECONE_AVAILABLE = False
    index = None

def verify_token(authorization: str = Header(...)):
    """Verify the authorization token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

def download_and_parse_pdf(pdf_url: str) -> str:
    """Download and parse PDF content from URL"""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        return text_content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split document into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks"""
    embeddings = []
    for text in texts:
        embedding = simple_embedding(text)
        embeddings.append(embedding)
    return embeddings

def store_embeddings(chunks: List[str], embeddings: List[List[float]]):
    """Store embeddings in Pinecone or locally"""
    if PINECONE_AVAILABLE and index:
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            })
        index.upsert(vectors=vectors)
    else:
        # Store locally in memory (for demo purposes)
        global local_chunks, local_embeddings
        local_chunks = chunks
        local_embeddings = embeddings
        print(f"📝 Stored {len(chunks)} chunks locally")

def search_relevant_clauses(query: str, top_k: int = 3) -> List[str]:
    """Search for relevant clauses using semantic search"""
    if PINECONE_AVAILABLE and index:
        query_embedding = simple_embedding(query)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        relevant_clauses = []
        for match in results.matches:
            relevant_clauses.append(match.metadata["text"])
        
        return relevant_clauses
    else:
        # Simple local search using keyword matching
        query_lower = query.lower()
        relevant_clauses = []
        
        for chunk in local_chunks:
            # Simple keyword-based relevance
            relevance_score = 0
            chunk_lower = chunk.lower()
            
            # Count matching words
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in chunk_lower:
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_clauses.append(chunk)
        
        # Return top chunks based on relevance
        return relevant_clauses[:top_k] if relevant_clauses else local_chunks[:top_k]

def generate_answer(query: str, relevant_clauses: List[str]) -> str:
    """Generate answer using Gemini LLM"""
    context = "\n\n".join(relevant_clauses)
    
    prompt = f"""
    Based on the following document clauses, answer the question accurately and concisely.
    
    Document Clauses:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process document queries and return answers
    """
    try:
        # Validate input
        if not request.documents or not request.questions:
            raise HTTPException(status_code=400, detail="Documents URL and questions are required")
        
        if len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Step 1: Download and parse PDF
        print(f"📄 Processing document: {request.documents}")
        document_text = download_and_parse_pdf(request.documents)
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the PDF")
        
        # Step 2: Chunk the document
        chunks = chunk_document(document_text)
        print(f"📝 Created {len(chunks)} chunks from document")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to process document content")
        
        # Step 3: Generate embeddings
        embeddings = generate_embeddings(chunks)
        print(f"🔢 Generated {len(embeddings)} embeddings")
        
        # Step 4: Store embeddings
        store_embeddings(chunks, embeddings)
        
        # Step 5: Process each question
        answers = []
        for i, question in enumerate(request.questions, 1):
            print(f"❓ Processing question {i}: {question[:50]}...")
            
            # Find relevant clauses
            relevant_clauses = search_relevant_clauses(question)
            print(f"📋 Found {len(relevant_clauses)} relevant clauses")
            
            # Generate answer using LLM
            answer = generate_answer(question, relevant_clauses)
            answers.append(answer)
            print(f"✅ Generated answer for question {i}")
        
        return QueryResponse(answers=answers)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error in process_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PolicyQuery AI is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PolicyQuery AI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 