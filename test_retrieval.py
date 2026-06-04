from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

query = "How long is an OAuth access token valid?"

results = vectorstore.similarity_search(
    query,
    k=5
)

for i, doc in enumerate(results):
    print(f"\n===== RESULT {i+1} =====\n")
    print(doc.page_content)