# Technical Summary

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers developer questions using the Upwork API documentation. The application uses LangChain for document processing, ChromaDB as the vector database, a Sentence Transformer embedding model for semantic search, DeepInfra-hosted Llama 3.1 for answer generation, and Streamlit for the user interface.

## Implementation Approach

1. The Upwork API documentation PDF is loaded and converted into text.
2. The extracted text is split into chunks of 500 characters with a 50-character overlap.
3. Each chunk is converted into vector embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.
4. Embeddings are stored locally in ChromaDB.
5. When a user submits a query, the system retrieves the most relevant document chunks using semantic similarity search.
6. The retrieved context is sent to the LLM along with a system prompt that enforces the role of a Senior Upwork API Consultant.
7. If the answer is not found in the retrieved context, the model returns a predefined hallucination-guard response.
8. The application displays the generated answer, retrieved source snippets, and response latency.

## Challenges Faced

* Extracting and preprocessing technical documentation from PDF format.
* Choosing an appropriate chunk size and overlap to preserve technical context.
* Ensuring relevant retrieval results from the vector database.
* Managing LLM latency while maintaining a responsive user experience.

## Use of LLM Assistance

* ChatGPT was used for learning and understanding RAG architecture.
* ChatGPT was used to understand LangChain, ChromaDB, and Streamlit integration.
* All generated code was reviewed, tested, and understood before inclusion in the final solution.

## Why I Am a Strong Candidate for ProAnalyst

1. I am able to learn new AI technologies quickly and apply them to practical problems.
2. I focus on building reliable, explainable, and maintainable solutions.
3. I am highly motivated to continue growing as an AI developer and contribute to real-world AI applications.
