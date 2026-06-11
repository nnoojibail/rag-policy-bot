from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Load the vector store we already built
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    openai_api_key=os.getenv("NEBIUS_API_KEY"),
    openai_api_base="https://api.studio.nebius.ai/v1",
    check_embedding_ctx_length=False,
    tiktoken_enabled=False
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up the LLM using Nebius
llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    openai_api_key=os.getenv("NEBIUS_API_KEY"),
    openai_api_base="https://api.studio.nebius.ai/v1"
)

# Prompt that forces cited answers
prompt = PromptTemplate.from_template("""Use the following context to answer the question.
Always cite which document your answer comes from.
If you don't know the answer from the context, say "I could not find this in the documents."
Do not make up answers.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 15 test questions
questions = [
    "What is the vacation policy?",
    "What is the code of conduct?",
    "What happens if an employee is terminated?",
    "What are the rules around conflicts of interest?",
    "What is the policy on workplace harassment?",
    "How do different companies handle employee ethics?",
    "What are the rules around gifts from vendors?",
    "What are the policies around outside employment?",
    "What happens to benefits when an employee leaves?",
    "What are the rules around confidential information?",
    "What is the parental leave policy?",
    "What is the remote work policy?",
    "Can employees date each other?",
    "What is the salary for a software engineer?",
    "What is the company's revenue last year?",
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {question}")
    print('='*60)
    answer = chain.invoke(question)
    print(answer)