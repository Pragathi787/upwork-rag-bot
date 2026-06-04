from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


# Step 1: Read PDF

reader = PdfReader("data/API Documentation Partial.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()


# Step 2: Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")


# Step 3: Embeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Store in Chroma

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector database created successfully.")