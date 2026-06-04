import os
import time

import streamlit as st

from dotenv import load_dotenv

from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


# DeepInfra client

client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)


# Embedding model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector DB

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


st.title("Upwork API Technical Support Bot")

query = st.text_input("Ask a question about the Upwork API")


if st.button("Submit") and query:

    start_time = time.time()

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

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

    for i, doc in enumerate(docs, 1):
        st.write(doc.page_content)
        st.divider()

    st.subheader("Latency")

    st.write(f"{latency:.2f} seconds")