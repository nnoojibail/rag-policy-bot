from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load all PDFs from the docs folder
docs_path = "docs"
all_documents = []

for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        print(f"Loading: {filename}")
        loader = PyPDFLoader(os.path.join(docs_path, filename))
        documents = loader.load()
        all_documents.extend(documents)

print(f"\nTotal pages loaded: {len(all_documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")
print("\n--- First 3 chunks ---\n")

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("---")