# RAG (Retrieval-Augmented Generation) with OpenAI, LangChain & Pinecone

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using OpenAI's language models, LangChain, and Pinecone. The goal is to retrieve relevant information from external documents (like PDFs) and generate accurate, context-aware answers.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Introduction

Retrieval-Augmented Generation (RAG) combines the power of large language models (like OpenAI's GPT-3.5) with external knowledge sources. Instead of relying on the model's internal memory, RAG retrieves relevant documents from a vector database and uses the language model to generate responses based on that retrieved content. This is especially useful for answering questions related to specialized or dynamic knowledge not contained within the model itself.

This project showcases how to implement a RAG pipeline that can extract, split, index, and query PDF documents to generate intelligent answers.

## Features

- **PDF Parsing and Text Splitting:** Load PDFs and break them into chunks for processing.
- **Vector Storage with Pinecone:** Store document embeddings in Pinecone for fast retrieval.
- **Question Answering:** Use OpenAI models to generate responses based on the retrieved content.
- **Retrieval-Augmented Generation:** Combines retrieval and generation to provide contextually accurate answers.

## Tech Stack

- **OpenAI API**: For generating responses using pre-trained large language models.
- **LangChain**: Helps manage and optimize large language models in the RAG pipeline.
- **Pinecone**: A scalable vector database for fast document retrieval.
- **PyPDFLoader**: A Python library for loading and parsing PDFs.
- **Jupyter Notebook**: The project is implemented as a Jupyter notebook for ease of use and demonstration.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- API keys for [OpenAI](https://openai.com/) and [Pinecone](https://www.pinecone.io/).

### Install Dependencies

```bash
pip install openai langchain pinecone-client pypdf2
```

### Configure Pinecone and OpenAI

Pinecone: Set up a Pinecone account and create an index. You’ll need the index name, namespace, and API key for the code. <br>
OpenAI: Get your OpenAI API key from the OpenAI platform.


### Project Structure
**Code Cells:** Each section of the notebook contains code for loading documents, splitting text, storing embeddings, and querying the database.
**Markdown Sections:** Describes the implementation and purpose of each part of the project.<br>
**RAG Implementation:** Follows a structured flow of data loading, embedding, indexing, and querying.

### Usage
**1. Load a PDF**
Use PyPDFLoader to load a PDF document.

```loader = PyPDFLoader("/path/to/your/pdf/file.pdf")
data = loader.load()
```

**2. Split Text into Chunks**
Use RecursiveCharacterTextSplitter to split the text into manageable chunks.

```text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_documents(data)
```

**3. Store Embeddings in Pinecone**
Create embeddings using OpenAI and store them in Pinecone.

```vectorstore_from_texts = PineconeVectorStore.from_texts(
    [f"Source: {t.metadata['source']}, Content: {t.page_content}" for t in texts],
    embeddings, index_name=index_name, namespace=namespace
)
```

**4. Query the System**
Once indexed, you can retrieve documents and generate answers.

```query = "What is the main idea of the first chapter?"
response = chain.run(input_documents=retrieved_docs, question=query)
print(response)
```
