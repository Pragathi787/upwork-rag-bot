import os
import time

import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


load_dotenv()

# DeepInfra Client

client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

# Read PDF

reader = PdfReader("data/API Documentation Partial.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Embedding model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings once at startup

chunk_embeddings = embedding_model.embed_documents(chunks)

st.title("Upwork API Technical Support Bot")

query = st.text_input("Ask a question about the Upwork API")

if st.button("Submit") and query:

    start_time = time.time()

    query_embedding = embedding_model.embed_query(query)

    similarities = []

    for i, emb in enumerate(chunk_embeddings):

        score = sum(
            a * b
            for a, b in zip(query_embedding, emb)
        )

        similarities.append(
            (score, chunks[i])
        )

    similarities.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top_chunks = similarities[:5]

    docs = [
        Document(page_content=chunk)
        for _, chunk in top_chunks
    ]

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    system_prompt = """
You are a Senior Upwork API Consultant.

Answer ONLY from the provided context.

If the answer is not found in the context, respond exactly:

I'm sorry, but the provided documentation does not contain that information.
"""

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Context:

{context}

Question:

{query}
"""
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    latency = time.time() - start_time

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for doc in docs:
        st.write(doc.page_content)
        st.divider()

    st.subheader("Latency")
    st.write(f"{latency:.2f} seconds")
