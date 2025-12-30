RAG-Based AI Assistant

Agentic AI Essentials – Final Project

***************************Overview***************************

This project implements a Retrieval-Augmented Generation (RAG) AI assistant that answers questions based on a custom document knowledge base.

Instead of relying only on a large language model’s internal knowledge, the assistant:

    *Searches a vector database of your documents
    
    *Retrieves the most relevant chunks
    
    *Uses those chunks as context to generate accurate, grounded answers

This approach improves factual accuracy, transparency, and domain specificity.

**************************Key Features**********************

Load custom text documents (.txt)

Automatically chunk documents for efficient retrieval

Generate embeddings using HuggingFace SentenceTransformers

Store and query embeddings using ChromaDB

Retrieve relevant context via semantic similarity search

Generate answers using OpenAI, Groq, or Google Gemini

Simple command-line interface (CLI)

Persistent vector database storage

*************************Project Architecture (RAG Pipeline)****************
User Question
      ↓
Query Embedding
      ↓
Vector Database (ChromaDB)
      ↓
Relevant Document Chunks
      ↓
Prompt + Context
      ↓
LLM Response

***************************Project Structure********************
rt-aaidc-project1-template/
├── src/
│   ├── app.py              # Main RAG application
│   ├── vectordb.py         # Vector DB + embedding logic
│   ├── documents/          # Knowledge base (.txt files)
│   └── chroma_db/          # Persistent vector store
├── requirements.txt
├── Pipfile
├── .env_example
├── README.md
└── LICENSE

*******************************Technologies Used************************

LangChain – Prompt & LLM orchestration

ChromaDB – Vector database

SentenceTransformers – Text embeddings

Groq / OpenAI / Google Gemini – LLM providers

Python 3.10+

*******************************Setup Instructions****************************
1. Clone the repository
git clone https://github.com/fahiyemuhammad/Agentic-AI-RAG-System.git
cd ./rt-aaidc-project1/


2. Install dependencies in virtual environment

python -m venv venv
source venv/bin/activate   # Linux / macOS
# OR
venv\Scripts\activate      # Windows

pip install -r requirements.txt

3. Configure environment variables
cp .env_example .env


Edit .env and add one API key:

GROQ_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here

Adding Your Own Documents

Place .txt files inside:

src/documents/


Each file should contain clean text related to the domain you want your assistant to answer questions about.

You can replace all existing files with company documents, research papers, manuals, or notes — the system will continue to work without code changes.

Running the Application

-First Run:   cd src
-Then run: python app.py


****************************Example interaction:******************

Ask a question (or 'quit'): What is Artificial Intelligence?

Answer:
Artificial Intelligence (AI) is a branch of computer science that focuses on building systems capable of performing tasks that typically require human intelligence...

*************************Example Queries************************

What is quantum computing?

Explain machine learning

What are the ethical concerns of AI?

How does deep learning work?

*************************How This Project Meets RAG Requirements**************

Document ingestion and chunking

Vector embeddings and storage

Semantic similarity search

Prompt → Retriever → LLM pipeline

Custom knowledge base

Working CLI interface

This satisfies the Agentic AI Essentials Certification project requirements.

********************Possible Future Enhancements ************************

Session-based memory

Agent-style reasoning (ReAct)

Web search tools

Evaluation metrics

UI frontend

Multi-agent orchestration (LangGraph / CrewAI)

***********************Final Notes****************************

This project demonstrates a fully functional RAG system, built using best practices in modern LLM application design.

It is intentionally modular so it can be extended into:

A company knowledge assistant

A documentation chatbot

A research paper explorer

Tested on Python 3.10 and 3.11