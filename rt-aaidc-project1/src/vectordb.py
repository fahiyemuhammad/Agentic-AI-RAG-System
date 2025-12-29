import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    Vector database wrapper using:
    - ChromaDB for storage
    - SentenceTransformers for embeddings
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database and embedding model.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace embedding model name
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Persistent ChromaDB storage
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model once (important for performance)
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, doc_id: int) -> List[Dict[str, Any]]:
        """
        Split a document into overlapping chunks.

        Why chunking?
        - Embeddings work best on small, focused text
        - Improves retrieval accuracy

        Args:
            text: Full document text
            doc_id: Index of the source document

        Returns:
            List of chunks with content and metadata
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_text(text)

        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source_doc": doc_id,
                        "chunk_index": i,
                    },
                }
            )

        return chunk_data

    def add_documents(self, documents: List[str]) -> None:
        """
        Ingest documents into the vector database.

        Pipeline:
        1. Chunk documents
        2. Embed chunks
        3. Store embeddings + text + metadata in ChromaDB
        """
        print(f"Processing {len(documents)} documents...")

        for doc_index, document in enumerate(documents):
            print(f"Chunking document {doc_index + 1}/{len(documents)}")

            # 1. Chunk document
            chunks = self.chunk_text(document, doc_index)

            # 2. Extract raw text
            texts = [chunk["content"] for chunk in chunks]

            # 3. Generate embeddings
            embeddings = self.embedding_model.encode(texts)

            # 4. Generate unique IDs
            ids = [f"doc_{doc_index}_chunk_{i}" for i in range(len(texts))]

            # 5. Store in vector DB
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[chunk["metadata"] for chunk in chunks],
                ids=ids,
            )

        print("Documents added to vector database successfully")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Perform similarity search against the vector database.

        Args:
            query: User query
            n_results: Number of chunks to retrieve

        Returns:
            ChromaDB search results dictionary
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query])

        # Perform vector search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Defensive handling for empty results
        if not results or not results.get("documents"):
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        return results
