from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


# =========================
# 1. LOAD PDF
# =========================

PDF_PATH = "documents/sample.pdf"

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Pages loaded: {len(pages)}")


# =========================
# 2. CREATE CHUNKS
# =========================

print("Creating chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(pages)

print(f"Chunks created: {len(chunks)}")


# =========================
# 3. CHECK METADATA
# =========================

print("Checking page metadata...")

for i, chunk in enumerate(chunks[:3], start=1):
    page_number = chunk.metadata.get("page", 0) + 1
    source = chunk.metadata.get("source", "unknown")

    chunk.metadata["page_number"] = page_number

    print(
        f"Chunk {i}: Page {page_number} | Source: {source}"
    )


# =========================
# 4. LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================
# 5. CREATE CHROMADB
# =========================

print("Creating ChromaDB vector store...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="oop_manual"
)

print("Documents stored in ChromaDB.")


# =========================
# 6. CREATE RETRIEVER
# =========================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# =========================
# 7. LOAD OLLAMA / LLAMA
# =========================

print("Loading Ollama model...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Ollama model loaded.")


# =========================
# 8. ASK QUESTION
# =========================

query = "What experiments are included in the OOP laboratory manual?"

print(f"\nQuery: {query}")

print("Retrieving relevant chunks...")

retrieved_docs = retriever.invoke(query)

print(f"Retrieved {len(retrieved_docs)} chunks.")


# =========================
# 9. BUILD CONTEXT
# =========================

context_parts = []

for i, doc in enumerate(retrieved_docs, start=1):

    page_number = doc.metadata.get(
        "page_number",
        doc.metadata.get("page", 0) + 1
    )

    source = doc.metadata.get(
        "source",
        "unknown"
    )

    context_parts.append(
        f"""
--- Source {i} ---
Page: {page_number}
Source: {source}

{doc.page_content}
"""
    )

context = "\n".join(context_parts)


# =========================
# 10. SEND CONTEXT TO LLAMA
# =========================

prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say that the information is not available in the document.

Keep the answer clear and concise.

Always mention the relevant page number when possible.

Context:
{context}

User Question:
{query}

Answer:
"""


print("\nGenerating answer using Llama...\n")

response = llm.invoke(prompt)


# =========================
# 11. DISPLAY FINAL ANSWER
# =========================

print("========== RAG ANSWER ==========")

print(response.content)

print("\n========== SOURCES ==========")

for i, doc in enumerate(retrieved_docs, start=1):

    page_number = doc.metadata.get(
        "page_number",
        doc.metadata.get("page", 0) + 1
    )

    source = doc.metadata.get(
        "source",
        "unknown"
    )

    print(
        f"{i}. {source} - Page {page_number}"
    )

print("\nRAG + ChromaDB + Ollama test completed successfully.")