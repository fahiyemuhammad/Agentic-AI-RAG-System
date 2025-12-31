import os
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader

from vectordb import VectorDB

# Load environment variables
load_dotenv()


def load_documents(documents_path: str) -> List[str]:
    """
    Load all .txt documents from a directory.

    Returns:
        List of document strings
    """
    documents = []

    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} documents")

    return [doc.page_content for doc in documents]


class RAGAssistant:
    """
    Retrieval-Augmented Generation assistant.
    """

    def __init__(self):
        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize vector database
        self.vector_db = VectorDB()

        # RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful research assistant.

Use the following context to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""
        )

        # Build LangChain pipeline
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized")

    def _initialize_llm(self):
        """
        Initialize LLM based on available API keys.
        Priority: Groq → OpenAI → Google
        """
        if os.getenv("GROQ_API_KEY"):
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.0,
            )

        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.0,
            )

        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
                temperature=0.0,
            )

        raise ValueError("No valid LLM API key found")

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector database.
        """
        self.vector_db.add_documents(documents)

    def invoke(self, question: str, n_results: int = 3) -> str:
        """
        Execute the full RAG pipeline.    
        Steps:
        1. Retrieve relevant chunks from the vector database
        2. Validate similarity scores to avoid weak / irrelevant context
        3. Build context from high-quality chunks
        4. Generate an answer using the LLM
        """

        # 1. Retrieve relevant chunks
        results = self.vector_db.search(question, n_results=n_results)
   
        documents = results.get("documents", [])
        distances = results.get("distances", [])
   
        # 2. Handle empty retrieval
        if not documents or not documents[0]:
            return "I could not find relevant information in the knowledge base."
   
        # 3. Distance-based filtering (important)
        # Lower distance = higher similarity
        SIMILARITY_THRESHOLD = 0.8
   
        filtered_chunks = []
        for doc, dist in zip(documents[0], distances[0]):
            if dist <= SIMILARITY_THRESHOLD:
                filtered_chunks.append(doc)
   
        # If nothing passes the similarity threshold, refuse to answer
        if not filtered_chunks:
            return (
                "I don't know. The knowledge base does not contain "
                "relevant information for this question."
            )
   
        # 4. Build context only from high-quality chunks
        context = "\n\n".join(filtered_chunks)
   
        # 5. Generate answer using the RAG chain
        return self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )



def main():
    assistant = RAGAssistant()

    docs = load_documents("./documents")
    assistant.add_documents(docs)

    while True:
        question = input("\nAsk a question (or type 'quit' to exit) ❔: ")
        if question.lower() == "quit":
            break

        answer = assistant.invoke(question)
        print("\n ✅ Answer:")
        print(answer)


if __name__ == "__main__":
    main()
