import uuid
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from chunker import TextChunker
from loaders import load_document
from llm_client import OpenRouterLLM

from logger import logger


class DocumentAssistant:
    """
    DocumentAssistant implements a simple RAG pipeline:
    - document loading
    - text chunking
    - embedding indexing
    - semantic search
    - answer generation via LLM
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        top_k: int = 3,
    ):
        self.chunker = TextChunker(chunk_size, overlap)
        self.top_k = top_k

        self.llm = OpenRouterLLM()

        self.chunks: List[str] = []
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


    def index_documents(self, documents: List[str]) -> None:
        """
        Loads documents, splits them into chunks,
        and computes embeddings for each chunk.
        """
        all_texts: List[str] = []

        for path in documents:
            text = load_document(path)
            all_texts.extend(text)
        self.chunks = self.chunker.split(all_texts)

        if not self.chunks:
            raise ValueError("No text chunks were created")

        self.vectorstore = FAISS.from_documents(
            self.chunks,
            self.embeddings
        )
        logger.info('Документы индексированы')

    def answer_query(self, query: str) -> str:
        """
        Finds relevant document chunks and generates
        an answer using an LLM.
        """
        if self.embeddings is None:
            raise RuntimeError("Documents are not indexed")
        if self.vectorstore is None:
            raise RuntimeError("Векторное хранилище не инициализировано. Вызовите index_documents сначала.")

        query_id = str(uuid.uuid4())
        query_embedding = self.embeddings.embed_query(query)

        docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
            query_embedding,
            k=self.top_k
        )

        retrieved_docs = [doc for doc, _ in docs_with_scores]

        retrieved_chunks = "\n\n".join(
            f"[Фрагмент {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        )

        prompt = self._build_prompt(query, retrieved_chunks)

        answer = self.llm.generate(prompt)
        logger.info({
            "event": "query_answered",
            "query_id": query_id,
            "query": query,
            "answer": answer
        })

        return answer, query_id
    
    def log_feedback(self, query_id: str, user_id: str, rating: int) -> None:
        if rating < 1 or rating > 5:
            raise ValueError("Оценка должна быть от 1 до 5")

        logger.info({
            "event": "user_feedback",
            "query_id": query_id,
            "user_id": user_id,
            "rating": rating
        })

    @staticmethod
    def _build_prompt(query: str, chunks: List[str]) -> str:
        context = "\n\n".join(chunks)

        return f"""Фрагменты документов:\n
            {context}\n\n
            Вопрос:\n
            {query}
            Ответ:"""
