# MediChat

This is a medical chatbot application that uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on medical documents. It leverages LangChain for orchestration, Pinecone for vector storage, and OpenAI's GPT-4o-mini (via OpenRouter) for generation.

## Features

- **Document Ingestion:** Automatically loads and processes PDF documents from the `data/` directory.
- **Vector Search:** Uses Pinecone to store and retrieve relevant document chunks based on semantic similarity using HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Generative AI:** Generates concise and accurate answers using GPT-4o-mini, grounded in the retrieved medical context.
- **Web Interface:** Includes a simple Flask-based web interface for chatting with the bot.

## Tech Stack

- **Backend:** Flask, Python
- **LLM Orchestration:** LangChain
- **Vector Database:** Pinecone
- **Embeddings:** HuggingFace (`sentence-transformers`)
- **LLM Provider:** OpenRouter (OpenAI GPT-4o-mini)

## Prerequisites

- Python 3.8 or higher
- A [Pinecone](https://www.pinecone.io/) account and API Key.
- An [OpenRouter](https://openrouter.ai/) API Key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ThiwankaChanditha/MediChat.git
    cd MediChat
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
HF_TOKEN=your_huggingface_token  # Optional if token is needed for specific models
```

## Usage

### 1. Indexing Documents

Before running the chat application, you need to ingest your medical PDF documents into the vector database.

1.  Place your PDF files in the `data/` directory.
2.  Run the indexing script:
    ```bash
    python store_index.py
    ```
    This script will:
    - Load PDFs from `data/`.
    - Split text into chunks.
    - Generate embeddings.
    - Create/Update the Pinecone index `medichat`.

### 2. Running the Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:8080`.

## Project Structure

```
MediChat/
├── app.py                 # Main Flask application
├── store_index.py         # Script to ingest and index data
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── src/
│   ├── helper.py          # Helper functions (PDF loading, splitting, embeddings)
│   └── prompt.py          # System prompts for the LLM
├── data/                  # Directory for source PDF files
├── templates/             # HTML templates for the web UI
└── static/                # Static assets (CSS, JS, images)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
