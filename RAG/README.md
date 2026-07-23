# PDF RAG Chatbot using LangChain + Ollama + Apache Solr

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents.

### Technologies Used

- Python
- LangChain
- Ollama
- Apache Solr
- Sentence Transformers
- PyPDF
- Jupyter Notebook

---

# Project Workflow

```
PDF
   │
   ▼
Load PDF
   │
   ▼
Chunk Documents
   │
   ▼
Generate Embeddings
   │
   ▼
Store in Apache Solr
   │
   ▼
User Question
   │
   ▼
Generate Question Embedding
   │
   ▼
Retrieve Similar Chunks
   │
   ▼
Ollama (LLM)
   │
   ▼
Answer
```

---

# Requirements

## Software

- Python 3.11+
- Java 17
- Apache Solr 9.x
- Ollama
- VS Code / Jupyter Notebook

---

# Project Structure

```
Project
│
├── documents/
│      sample.pdf
│
├── notebook.ipynb
│
├── requirements.txt
│
└── README.md
```

---

# Install Java

Verify Java installation

```powershell
java -version
```

Example

```
openjdk version "17.x.x"
```

---

# Apache Solr

## Go to Solr Folder

```powershell
cd C:\solr-9.10.1\bin
```

(Replace with your Solr installation path.)

---

# Check Solr Version

```powershell
.\solr.cmd version
```

---

# Start Solr

```powershell
.\solr.cmd start
```

or

```powershell
.\solr.cmd start -p 8983
```

---

# Check Solr Status

```powershell
.\solr.cmd status
```

---

# Open Solr Dashboard

Open your browser

```
http://localhost:8983
```

---

# Stop Solr

Stop Solr running on port 8983

```powershell
.\solr.cmd stop -p 8983
```

Stop all Solr instances

```powershell
.\solr.cmd stop --all
```

---

# Restart Solr

```powershell
.\solr.cmd restart
```

or

```powershell
.\solr.cmd restart -p 8983
```

---

# Create a Solr Core

```powershell
.\solr.cmd create -c rag_pdf
```

where

```
rag_pdf
```

is the core name.

---

# Delete a Solr Core

```powershell
.\solr.cmd delete -c rag_pdf
```

---

# List Available Cores

Open

```
http://localhost:8983/solr/admin/cores
```

or use

```powershell
.\solr.cmd status
```

---

# Check if Solr is Running

PowerShell

```powershell
Test-NetConnection localhost -Port 8983
```

Expected

```
TcpTestSucceeded : True
```

---

# Ollama

## Check Installation

```powershell
ollama --version
```

---

# List Installed Models

```powershell
ollama list
```

---

# Download a Model

Example

```powershell
ollama pull qwen2.5:3b
```

---

# Run Model

```powershell
ollama run qwen2.5:3b
```

Exit

```
/bye
```

---

# Remove a Model

```powershell
ollama rm qwen2.5:3b
```

---

# Python Environment

Create Virtual Environment

```powershell
python -m venv .venv
```

Activate

```powershell
.venv\Scripts\activate
```

---

# Install Packages

```powershell
pip install -r requirements.txt
```

or

```powershell
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install sentence-transformers
pip install pysolr
pip install pypdf
pip install jupyter
```

---

# Start Jupyter

```powershell
jupyter notebook
```

---

# Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Vector Dimension

```
384
```

---

# LLM

Current

```
qwen2.5:3b
```

---

# Solr Schema

Fields

| Field | Type |
|---------|------|
| id | string |
| content | text_general |
| source | string |
| page | pint |
| vector | DenseVectorField |

Vector dimension

```
384
```

Similarity

```
cosine
```

---

# Chunking Strategy

Current implementation uses

```
RecursiveCharacterTextSplitter
```

Typical configuration

```
Chunk Size      : 400

Chunk Overlap   : 50
```

---

# Retrieval

The project uses

```
KNN Vector Search
```

Example

```
Top K = 2
```

---

# Conversation Memory

Implemented using

```
RunnableWithMessageHistory
```

Stores previous user questions and assistant responses.

---

# Response Cache

The project includes

```
Exact Question Cache
```

Repeated questions return immediately without

- Embedding generation
- Solr search
- LLM generation

---

# Current Features

- PDF Loading
- Recursive Chunking
- Embedding Generation
- Apache Solr Vector Database
- KNN Search
- Ollama Integration
- Conversation Memory
- Response Cache
- Source Citation

---

# Future Improvements

- Multi-PDF Support
- Hybrid Search
- Semantic Cache
- History-aware Retrieval
- Cross Encoder Reranker
- Streaming Responses
- Agent Support
- FastAPI Deployment

---

# Troubleshooting

## Solr Not Running

Check

```powershell
.\solr.cmd status
```

If stopped

```powershell
.\solr.cmd start -p 8983
```

---

## Port Already in Use

```
Address already in use
```

Stop existing Solr

```powershell
.\solr.cmd stop --all
```

---

## Java Not Found

Check

```powershell
java -version
```

Set

```
JAVA_HOME
```

to Java 17.

---

## Ollama Not Running

Start Ollama

```powershell
ollama serve
```

---

## Check Installed Models

```powershell
ollama list
```

---

# How to Run the Project

1. Start Solr

```powershell
.\solr.cmd start
```

2. Verify Solr

```
http://localhost:8983
```

3. Verify Ollama

```powershell
ollama list
```

4. Open the notebook

```
notebook.ipynb
```

5. Run all cells

6. Ask questions

Example

```
What is a wellhead?

Explain drilling.

What are the functions of a separator?
```

---

# Author

Abid Dillep Jan