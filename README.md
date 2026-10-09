# DocInsight

A simple Retrieval Augmented Generation (RAG) application that allows users to upload PDF and text files, index their contents, and perform semantic searches or ask questions based on the uploaded documents.

## Features

- Upload PDF and text files
- Process and index document content
- Ask questions about your documents
- Get answers with source references
- Clean and intuitive user interface

## Technology Stack

- **LlamaIndex**: Main framework for document indexing and retrieval
- **ChromaDB**: Vector database for storing and searching document embeddings
- **Streamlit**: Frontend framework for the user interface
- **OpenAI API**: For generating embeddings and LLM responses

## Implementation Details

### Document Processing

Documents are processed through the following workflow:
1. Files are loaded using LlamaIndex's SimpleDirectoryReader
2. Each document gets a stable id derived from a hash of the file's contents
3. Text is chunked once, using SentenceSplitter with configurable chunk size and overlap
4. Chunks from documents not already in the index are embedded with OpenAI's embedding model
5. The chunks and their embeddings are stored in ChromaDB, which persists across restarts

### Retrieval and Response Generation

When a user asks a question:
1. The query is processed through the VectorStoreIndex to retrieve relevant document chunks
2. Retrieved chunks are used as context for the OpenAI LLM (gpt-4o-mini)
3. The model generates a response based on the provided context
4. Both the answer and source references are displayed to the user

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key

### Installation

#### Using Docker (Recommended)

1. Clone this repository:
   ```
   git clone https://github.com/arjunurs/docinsight.git
   cd docinsight
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env with your actual API key
   ```

3. Build and run the Docker container:
   ```
   docker-compose up -d
   ```

4. Access the application at http://localhost:8501

#### Manual Installation

1. Clone this repository:
   ```
   git clone https://github.com/arjunurs/docinsight.git
   cd docinsight
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env with your actual API key
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

6. Access the application at http://localhost:8501

## Usage

1. Upload one or more PDF or text files
2. Click "Process Documents" to index them
3. Enter your question in the text field
4. Click "Ask" to get answers based on your documents

## Configuration

All settings are optional environment variables, read at startup (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlapping tokens between chunks |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL` | gpt-4o-mini | OpenAI model used to answer |
| `LLM_TEMPERATURE` | 0.0 | Kept at 0 so answers stay grounded and repeatable |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` only affects newly indexed documents; clear `data/chroma` to re-index everything.

Indexed documents persist in `data/chroma` and are available again after a restart. Each document is identified by a hash of its contents, so uploading the same file again does not create duplicate entries.

## Running Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests use a mock embedding model and make no OpenAI calls.

## Troubleshooting

If you encounter any issues:

1. Make sure your OpenAI API key is correctly set in the .env file
2. For Docker deployment, ensure ports 8501 is not already in use
3. If you see errors related to document processing, check that your PDF files are text-based and not scanned images
4. Restart the application with `docker-compose down && docker-compose up -d` if needed

## Future Enhancements

### Document Handling
- Support for additional file formats (DOCX, HTML)
- Metadata extraction from documents
- Document chunking optimization
- Multi-modal support (images, audio)

### Search & Retrieval
- Hybrid search capabilities (keyword + semantic)
- Advanced filtering and structured queries
- API integration with external knowledge sources

### User Experience
- Conversation memory with follow-up question handling
- Chat-like interface with history
- Responsive design

