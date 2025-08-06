# PolicyQuery AI - Project Structure

## 📁 File Organization

```
PolicyQuery AI/
├── 📄 main.py                    # Main FastAPI application (basic version)
├── 📄 main_enhanced.py           # Enhanced version with modular architecture
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                  # Comprehensive documentation
├── 📄 setup.py                   # Automated setup script
├── 📄 test_api.py                # API testing script
├── 📄 config.py                  # Configuration management
├── 📄 utils.py                   # Utility classes and functions
├── 📄 env_example.txt            # Environment variables template
├── 📄 .gitignore                 # Git ignore rules
└── 📄 PROJECT_STRUCTURE.md       # This file

```

## 🏗️ Architecture Overview

### Core Components

#### 1. **Main Applications**

- **`main.py`**: Basic FastAPI implementation with all functionality in one file
- **`main_enhanced.py`**: Modular architecture with separate utility classes

#### 2. **Configuration Management**

- **`config.py`**: Centralized configuration with environment-specific settings
- **`env_example.txt`**: Template for environment variables

#### 3. **Utility Classes (`utils.py`)**

- **`DocumentProcessor`**: Handles document chunking and text cleaning
- **`EmbeddingManager`**: Manages embedding generation using SentenceTransformers
- **`VectorStoreManager`**: Handles Pinecone vector database operations
- **`LLMManager`**: Manages Google Gemini LLM interactions
- **`PerformanceMonitor`**: Tracks and reports performance metrics

#### 4. **Testing & Validation**

- **`test_api.py`**: Comprehensive API testing script
- **`setup.py`**: Automated setup and validation

## 🔄 Workflow

### Six-Step Process

1. **Input Documents** (`main.py:download_and_parse_pdf()`)

   - Receives PDF blob URLs
   - Downloads and parses PDF content
   - Validates document size and format

2. **Document Processing** (`utils.py:DocumentProcessor`)

   - Cleans and normalizes text
   - Creates overlapping chunks
   - Adds metadata for tracking

3. **Embedding Search** (`utils.py:EmbeddingManager`)

   - Generates embeddings using all-MiniLM-L6-v2
   - Stores vectors in Pinecone database
   - Enables semantic search capabilities

4. **Clause Matching** (`utils.py:VectorStoreManager`)

   - Performs semantic search for relevant clauses
   - Returns top-k most similar chunks
   - Provides explainability through clause retrieval

5. **Logic Evaluation** (`utils.py:LLMManager`)

   - Uses Gemini Pro for answer generation
   - Provides contextual responses
   - Handles error cases gracefully

6. **JSON Output** (`main.py:process_query()`)
   - Returns structured JSON responses
   - Includes performance metrics
   - Maintains question-answer order

## 🔧 Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional
PINECONE_ENVIRONMENT=gcp-starter
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Model Configuration

- **LLM**: Google Gemini Pro
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: Pinecone with cosine similarity
- **Chunk Size**: 1000 words with 200 word overlap

## 🚀 Getting Started

### Quick Start

```bash
# 1. Run setup script
python setup.py

# 2. Edit .env file with API keys

# 3. Run application
python main.py
# or
python main_enhanced.py

# 4. Test API
python test_api.py
```

### Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
cp env_example.txt .env
# Edit .env with your API keys

# 3. Run application
python main.py

# 4. Access API
curl http://localhost:8000/health
```

## 📊 Performance Features

### Optimization Strategies

- **Token Efficiency**: Smart chunking with overlap
- **Low Latency**: Efficient vector search
- **Scalability**: Modular architecture
- **Explainability**: Clause traceability

### Monitoring

- **Performance Metrics**: Built-in timing for each operation
- **Health Checks**: Comprehensive service monitoring
- **Error Handling**: Graceful failure management
- **Logging**: Detailed operation tracking

## 🔒 Security Features

### Authentication

- **Bearer Token**: Secure API access
- **Input Validation**: Comprehensive request validation
- **Error Sanitization**: Safe error messages
- **API Key Management**: Secure environment variable handling

### Data Protection

- **Text Sanitization**: Removes potentially dangerous characters
- **Size Limits**: Prevents oversized document processing
- **URL Validation**: Ensures valid document URLs
- **Request Limits**: Prevents abuse through validation rules

## 🧪 Testing

### Test Coverage

- **Health Endpoints**: Basic service availability
- **Authentication**: Token validation
- **API Functionality**: Complete workflow testing
- **Error Handling**: Invalid request testing
- **Performance**: Response time monitoring

### Test Commands

```bash
# Run all tests
python test_api.py

# Individual test functions available in test_api.py
```

## 📈 Scalability Considerations

### Horizontal Scaling

- **Stateless Design**: No session dependencies
- **External Services**: Pinecone and Gemini handle scaling
- **Load Balancing**: Ready for multiple instances

### Performance Tuning

- **Chunk Size**: Configurable for different document types
- **Search Parameters**: Adjustable top-k values
- **Timeout Settings**: Configurable request timeouts
- **Concurrency**: Built-in request limiting

## 🔄 Deployment

### Development

```bash
python main.py
# or
uvicorn main:app --reload
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# or
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Future Enhancement)

```dockerfile
# Dockerfile would be added for containerization
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]
```

## 📚 API Documentation

### Interactive Docs

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Endpoints

- **POST /api/v1/hackrx/run**: Main query processing
- **GET /**: Health check
- **GET /health**: Detailed health status
- **GET /config**: Configuration info (debug mode)

## 🛠️ Development

### Code Quality

- **Modular Design**: Separated concerns
- **Type Hints**: Full type annotation
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Graceful failure management

### Extensibility

- **Plugin Architecture**: Easy to add new document types
- **Model Swapping**: Configurable LLM and embedding models
- **Vector Store**: Pluggable vector database
- **Custom Validators**: Extensible request validation

## 🎯 Hackathon Features

### Evaluation Criteria Alignment

- **Reusability**: Modular, well-documented code
- **Explainability**: Clause traceability and decision rationale
- **Optimization**: Token efficiency and low latency
- **Accuracy**: Contextual answer generation
- **Scalability**: Production-ready architecture

### Real-World Applicability

- **Insurance Domain**: Policy document processing
- **Legal Domain**: Contract clause analysis
- **HR Domain**: Policy and procedure queries
- **Compliance Domain**: Regulatory document analysis

---

**Built with ❤️ for PolicyQuery AI - Bajaj Finserv Hackathon**
