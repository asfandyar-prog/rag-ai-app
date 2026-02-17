# 🧠 RAG AI App

A Retrieval-Augmented Generation (RAG) system that enables users to chat with their own PDF documents using semantic search and LLM-powered reasoning.

Built using **Streamlit, LangChain, Sentence Transformers, ChromaDB, and Groq (LLaMA 3)**.

---

## 🚀 Overview

RAG AI App allows intelligent question-answering over custom documents by combining:

- Vector embeddings
- Semantic similarity search
- Context retrieval
- Large Language Model generation

Instead of relying on general knowledge, the model answers strictly based on the retrieved document context.

---

## ✨ Features

- 📄 Multi-PDF ingestion
- ✂️ Intelligent document chunking
- 🧠 Embeddings using `all-MiniLM-L6-v2`
- 🗄️ Persistent vector storage with ChromaDB
- 🔍 Top-k semantic retrieval
- 🤖 LLM response generation using Groq (LLaMA 3)
- 💬 Interactive Streamlit chat interface
- 🌙 Clean dark professional UI

---

## 🏗️ System Architecture
User Query
↓
Query Embedding
↓
Vector Similarity Search (ChromaDB)
↓
Top-k Relevant Chunks
↓
Context Construction
↓
Groq LLaMA 3
↓
Grounded Answer

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Sentence Transformers
- ChromaDB
- Groq API (LLaMA 3)
- PyMuPDF
- Scikit-learn

---

## 📂 Project Structure
User Query
↓
Query Embedding
↓
Vector Similarity Search (ChromaDB)
↓
Top-k Relevant Chunks
↓
Context Construction
↓
Groq LLaMA 3
↓
Grounded Answer


---

## ⚙️ Local Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/asfandyar-prog/rag-ai-app.git
cd rag-ai-app
2️⃣ Create a virtual environment
python -m venv .venv


Activate it:

Windows

.venv\Scripts\activate


Mac/Linux

source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variable

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

5️⃣ Run the application
streamlit run apps.py

🔐 Environment Variables
Variable	Required	Description
GROQ_API_KEY	Yes	API key for Groq LLaMA 3 inference
🧠 How It Works

PDFs are loaded using PyMuPDF.

Documents are split into overlapping chunks.

Each chunk is converted into vector embeddings.

Embeddings are stored in ChromaDB.

User query is embedded.

Top-k similar chunks are retrieved.

Retrieved context is sent to Groq LLaMA 3.

The LLM generates a grounded answer.

🎯 Use Cases

Research paper Q&A

Resume and proposal review

Academic assistant

Knowledge base chatbot

Personal document intelligence

🔮 Future Improvements

Reranking layer for better retrieval accuracy

Source citation highlighting

File upload support in UI

Multi-source ingestion (PDF + text)

Production deployment optimization

👨‍💻 Author

Asfand Yar
BSc Computer Science 
Interested in Machine Learning, AI Systems, and Quantum Computing.

