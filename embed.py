from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Load all PDFs
docs_path = "docs"
all_documents = []

for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        print(f"Loading: {filename}")
        loader = PyPDFLoader(os.path.join(docs_path, filename))
        documents = loader.load()
        all_documents.extend(documents)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"Total chunks to embed: {len(chunks)}")

# Embed using Nebius and store in ChromaDB
print("\nEmbedding and storing in ChromaDB... (this may take a minute)")

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    openai_api_key=os.getenv("NEBIUS_API_KEY"),
    openai_api_base="https://api.studio.nebius.ai/v1",
    check_embedding_ctx_length=False,
    tiktoken_enabled=False
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"\n✅ Done! {len(chunks)} chunks embedded and stored in ./chroma_db")