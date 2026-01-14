# RAG-Based AI Assistant

**Agentic AI Essentials – Module 1 Final Project**

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** AI assistant that answers questions based strictly on a custom document knowledge base.

Rather than relying solely on a language model’s internal knowledge, the system:
- Retrieves relevant information from a vector database
- Injects retrieved context into the prompt
- Generates grounded, context-aware responses

This approach improves factual accuracy, transparency, and domain specificity.

## Key Features

- Custom document ingestion (`.txt` files)
- Automatic document chunking for retrieval efficiency
- Semantic embeddings using **HuggingFace SentenceTransformers**
- Vector storage and similarity search via **ChromaDB**
- Threshold-based retrieval filtering with fallback logic
- **Short-term conversational memory** for multi-turn interactions
- Support for multiple LLM providers (Groq, OpenAI, Google Gemini)
- Simple command-line interface (CLI)
- Persistent vector database storage

## Conversational Memory

The assistant includes **short-term conversational memory** by storing recent user–assistant exchanges and injecting them into the prompt at inference time.

- Enables follow-up questions and contextual continuity (e.g., “Tell me more about it” works after “What is AI?”)
- Memory window is bounded to control token usage
- Prevents hallucination by maintaining strict grounding in retrieved context

This prompt-based memory strategy provides a practical baseline for conversational RAG systems.

## Project Architecture (RAG Pipeline)
User Query
↓
Query Embedding (with optional rewriting for follow-ups)
↓
Vector Database (ChromaDB)
↓
Relevant Document Chunks (filtered by similarity + fallback)
↓
Prompt (Context + Memory)
↓
LLM → Grounded Response
text## Project Structure
rt-aaidc-project1/
├── src/
│   ├── app.py                  # Main RAG application (with memory)
│   ├── vectordb.py             # Vector DB and embedding logic
│   ├── documents/              # Knowledge base (.txt files)
│   └── chroma_db/              # Persistent vector store (auto-created)
├── requirements.txt
├── .env_example
├── README.md
└── LICENSE
text## Technologies Used

- LangChain – Prompt orchestration and LLM abstraction
- ChromaDB – Vector database
- SentenceTransformers – Text embeddings
- Groq / OpenAI / Google Gemini – LLM providers
- Python 3.10+

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/fahiyemuhammad/Agentic-AI-RAG-System.git
cd rt-aaidc-project1
2. Create Virtual Environment & Install Dependencies
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

 pip install --upgrade pip
 pip install -r requirements.txt
Installation Note
The first install may take 15–40 minutes due to large dependencies (e.g., PyTorch CPU). This is normal and only happens once.
3. Configure Environment Variables
 cp .env_example .env
Edit .env and add at least one API key:
envGROQ_API_KEY=your_groq_key_here
# OR
OPENAI_API_KEY=your_openai_key_here
# OR
GOOGLE_API_KEY=your_google_key_here
4. Add Your Own Documents (Optional)
Place any .txt files in src/documents/.
The assistant will automatically load and index them.
Running the Application
 cd src
 python app.py
Example Interaction (with memory)
textAsk a question (or 'quit' to exit):

You: What is Artificial Intelligence?
Assistant: Based on available information: Artificial Intelligence (AI) is a branch of computer science...

You: Tell me more about its history.
Assistant: Based on available information: The history of AI is not covered in detail in the provided context...

You: And what are its main ethical concerns?
Assistant: Based on available information: The ethics of AI include concerns about fairness, avoiding bias...
Maintenance & Support

Maintenance: This is a personal project for the Agentic AI Essentials certification. The code is intentionally simple and modular for easy extension.
Support: Issues can be reported via GitHub Issues on the repository.
Future: Planned enhancements include long-term memory, web UI, and quantitative evaluation metrics.

Evaluation Considerations
Current evaluation is based on:

Retrieval hit relevance (top-K similarity)
Context grounding
Hallucination avoidance
Multi-turn conversational coherence

Future improvements could include quantitative metrics (precision@k, recall@k, RAGAS scores).
Final Notes
This project demonstrates a production-style RAG system with modular design, conversational memory, and grounded reasoning.
It is structured to support incremental extension into:

Company knowledge assistants
Documentation chatbots
Research exploration tools

Tested on Python 3.10 and 3.11.
Thank you for reviewing!