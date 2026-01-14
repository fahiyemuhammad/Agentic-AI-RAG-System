import os
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader

from vectordb import VectorDB

load_dotenv()


def load_documents(documents_path: str = "./documents") -> List[str]:
    """Load all .txt documents from a directory."""
    documents = []
    if not os.path.exists(documents_path):
        print(f"Warning: Documents path '{documents_path}' not found.")
        return documents

    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} documents")
    return [doc.page_content for doc in documents]


class RAGAssistant:
    """RAG assistant with conversation memory and improved retrieval."""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.vector_db = VectorDB()

        # Conversation history (short-term memory)
        self.chat_history = []

        # Main prompt with memory support
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant.

Answer using ONLY the provided context and previous conversation.
If the context contains relevant information (even partial), use it and start with "Based on available information:".
If nothing relevant is found, say "I don't have enough information in the knowledge base to answer accurately."
Do NOT use outside knowledge or make up facts.

Context:
{context}

Previous conversation:
{chat_history}

Current question:
{question}

Answer:"""),
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized (with memory)")

    def _initialize_llm(self):
        if os.getenv("GROQ_API_KEY"):
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.3,
            )
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
            )
        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
                temperature=0.3,
            )
        raise ValueError("No valid LLM API key found")

    def add_documents(self, documents: List[str]) -> None:
        self.vector_db.add_documents(documents)

    def invoke(self, question: str, n_results: int = 6) -> str:
        # Retrieve chunks
        results = self.vector_db.search(question, n_results=n_results)
        documents = results.get("documents", [])
        distances = results.get("distances", [])

        if not documents or not documents[0]:
            return "I don't have enough information in the knowledge base to answer this."

        # Looser threshold
        GOOD_THRESHOLD = 0.55
        FALLBACK_THRESHOLD = 0.85

        filtered_chunks = []
        for doc, dist in zip(documents[0], distances[0]):
            if dist <= GOOD_THRESHOLD:
                filtered_chunks.append(doc)

        # Fallback: take top 2 if top result is reasonable
        if not filtered_chunks and documents and documents[0]:
            if distances[0][0] <= FALLBACK_THRESHOLD:
                filtered_chunks = documents[0][:2]
            else:
                return "I don't have enough relevant information to answer accurately."

        context = "\n\n".join(filtered_chunks)

        # Format history (last 8 turns)
        history_text = ""
        for msg in self.chat_history[-8:]:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"

        # Generate answer
        answer = self.chain.invoke({
            "context": context,
            "chat_history": history_text,
            "question": question,
        })

        # Save to memory
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer


def main():
    assistant = RAGAssistant()

    docs = load_documents()
    if docs:
        assistant.add_documents(docs)

    print("\nRAG Assistant ready! (Memory-enabled)")
    print("Ask anything (or 'quit' to exit):\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("Assistant:", end=" ", flush=True)
        answer = assistant.invoke(question)
        print(answer)


if __name__ == "__main__":
    main()