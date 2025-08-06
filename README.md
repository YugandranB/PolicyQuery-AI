# PolicyQuery AI

An LLM-Powered Intelligent Query-Retrieval System for processing large documents and making contextual decisions. Built for the Bajaj Finserv national-level hackathon.

## 🚀 Features

- **Document Processing**: Handles PDF, DOCX, and email documents
- **Semantic Search**: Uses Pinecone vector database for efficient clause retrieval
- **LLM Integration**: Powered by Google Gemini for intelligent answer generation
- **RESTful API**: FastAPI-based with comprehensive documentation
- **Authentication**: Secure Bearer token authentication
- **Explainability**: Provides clear decision rationale and clause traceability

## 🏗️ System Architecture

The system follows a six-step workflow:

1. **Input Documents**: Receives PDF blob URLs and natural language queries
2. **Document Processing**: Downloads and parses PDF content
3. **Embedding Search**: Chunks documents and generates embeddings
4. **Clause Matching**: Performs semantic search to find relevant clauses
5. **Logic Evaluation**: Uses Gemini LLM to generate contextual answers
6. **JSON Output**: Returns structured responses with answers

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- Pinecone API key
- Internet connection for API calls

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd PolicyQuery-AI
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   ```bash
   # Copy the example environment file
   cp env_example.txt .env

   # Edit .env with your API keys
   GEMINI_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=gcp-starter
   ```

## 🚀 Running the Application

### Development Mode

```bash
python main.py
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## 📚 API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

### Authentication

All requests require Bearer token authentication:

```
Authorization: Bearer 3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc
```

### Endpoints

#### POST /hackrx/run

Process document queries and return answers.

**Request Body**:

```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHN%CE%96%CE%97zQ%3D",
  "questions": [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
  ]
}
```

**Response Body**:

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
  ]
}
```

#### GET /

Health check endpoint.

#### GET /health

Detailed health check endpoint.

## 🔧 Configuration

### Environment Variables

| Variable               | Description           | Default       |
| ---------------------- | --------------------- | ------------- |
| `GEMINI_API_KEY`       | Google Gemini API key | Required      |
| `PINECONE_API_KEY`     | Pinecone API key      | Required      |
| `PINECONE_ENVIRONMENT` | Pinecone environment  | `gcp-starter` |
| `HOST`                 | Server host           | `0.0.0.0`     |
| `PORT`                 | Server port           | `8000`        |

### Model Configuration

- **LLM**: Google Gemini Pro
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: Pinecone with cosine similarity
- **Chunk Size**: 1000 words with 200 word overlap

## 🧪 Testing

### Using curl

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHN%CE%96%CE%97zQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
  }'
```

### Using Python requests

```python
import requests
import json

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHN%CE%96%CE%97zQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print(json.dumps(response.json(), indent=2))
```

## 📊 Performance Optimization

The system is optimized for:

- **Token Efficiency**: Smart chunking and embedding strategies
- **Low Latency**: Efficient vector search and caching
- **Scalability**: Modular architecture for easy scaling
- **Explainability**: Clear decision rationale and clause traceability

## 🔒 Security

- Bearer token authentication
- Input validation and sanitization
- Error handling without information leakage
- Secure API key management

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Gemini and Pinecone API keys are correctly set in the `.env` file
2. **PDF Processing Errors**: Check if the PDF URL is accessible and the file is not corrupted
3. **Memory Issues**: For large documents, consider reducing chunk size
4. **Network Issues**: Ensure stable internet connection for API calls

### Logs

The application provides detailed logging for debugging:

- PDF processing status
- Embedding generation progress
- Search query results
- LLM response generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed for the Bajaj Finserv hackathon.

## 🆘 Support

For support and questions:

- Check the API documentation at `http://localhost:8000/docs`
- Review the troubleshooting section
- Open an issue in the repository

---

**Built with ❤️ for PolicyQuery AI**
