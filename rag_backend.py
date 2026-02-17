

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf"

print("Looking for PDFs in:", PDF_DIR)

if not PDF_DIR.exists():
    raise FileNotFoundError(f"PDF directory not found at {PDF_DIR}")

loader = DirectoryLoader(
    str(PDF_DIR),
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

docs = loader.load()


def read_all_docs(directory):
    """Read all the documents from the given directory"""
    all_docs = []
    pdf_dir = Path(directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    for pdf_file in pdf_files:
        print(f"pdf {pdf_file.name} is being processed")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
                all_docs.append(doc)
        except Exception as e:
            print(f"error while processing pdf {pdf_file.name} {e}")
            raise

    return all_docs  


doc = read_all_docs("../GenerativeAI/pdf")



# Make chunks

def make_chunks(document, chunk_size=2000, chunk_overlap=200):
    """Makes chunks of the documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ',', ' ', ''],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    doc_splitter = text_splitter.split_documents(document)
    if doc_splitter:
        print("documents are splitted in chunks")
    return doc_splitter


split = make_chunks(doc)



# Embeddings

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict, Any

class Embeddings:
    """handle the documents and make the embeddings of document's chunks"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Load the model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"model is successfully loaded {self.model_name}")
        except Exception as e:
            print(f"error while loading the model {e}")
            raise

    def get_embeddings(self, texts: List[str]):
        """Takes the list of chunks and make the embeddings for each chunk."""
        if not self.model:
            raise RuntimeError("Model not loaded.")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error occured while processing the embeddings of chunks {e}")
            raise


#  embed CHUNKS, not full docs
texts = [d.page_content for d in split]
embeddings_model = Embeddings()
chunk_embeddings = embeddings_model.get_embeddings(texts)



# Vector Store

class Vectors:
    """Handle the embeddings and store it in the vector base"""
    def __init__(self, collection_name="rag_collection", persistent_dir="./multi_vector_store"):
        self.collection_name = collection_name
        self.persistent_dir = persistent_dir
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize the vector store"""
        os.makedirs(self.persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persistent_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF documents embeddings for RAG"}
        )
        print(f"Initialized VectorStore with collection: {self.collection_name}")

    def add_documents(self, documents, embeddings: np.ndarray):
        """Add documents and their embeddings to the vector base"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        embeddings_list = []
        documents_text = []
        metadatas = []

        #  fix enumerate(zip(...)) unpack
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)  #  ids not id

            metadata = dict(doc.metadata)  #  dict(...) not Dict(...)
            metadata["index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(emb.tolist())  #  each vector, not whole array

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Added {len(documents)} docs with {len(embeddings_list)} embeddings")
        except Exception as e:
            print("error occured while handling the document and embedding storing")
            raise

    def query(self, query_embeddings: np.ndarray, top_k: int = 5):
        """handle the query and make its embeddings"""
        try:
            # chroma expects list[list[float]]
            if hasattr(query_embeddings, "tolist"):
                query_embeddings = query_embeddings.tolist()

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,  #  n_results not top_k
                include=["documents", "distances", "metadatas"]  # correct keys
            )
            return results
        except Exception as e:
            print(f"error while handling the embeddings of query. {e}")
            raise


vectorstore = Vectors()
vectorstore.add_documents(split, chunk_embeddings)



# Retriever

class RetrieverAugmentedGeneration:
    def __init__(self, vectorstore: Vectors, embeddings: Embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.01) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.get_embeddings([query])  # (1, dim)

        results = self.vectorstore.collection.query(
            query_embeddings=query_embedding.tolist(),  # 
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )

        retrieved_docs = []
        for i, (doc_id, distance, doc_text, metadata) in enumerate(
            zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ):
            similarity = 1.0 - distance
            print(f"Rank {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")

            if similarity >= threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "documents": doc_text,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "distance": distance,
                    "rank": i + 1,
                })

        return retrieved_docs



# LLM + Pipeline
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

retriever = RetrieverAugmentedGeneration(vectorstore, embeddings_model)


def reg_simple(query, retriever, llm, top_k=3):
    # retrieve context
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([d["documents"] for d in results]) if results else "No context found"  # ✅ documents key

    prompt = f"""Answer the following question based on the context provided.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt) 
    return response.content
