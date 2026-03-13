import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from models.embeddings import get_embedding_model
from config.config import CHROMA_PERSIST_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS


def load_and_split_pdfs(data_dir: str = DATA_DIR):
    """Load all PDFs from data/ and split into chunks."""
    docs = []
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                path = os.path.join(data_dir, filename)
                loader = PyPDFLoader(path)
                pages = loader.load()
                # Tag each page with source filename
                for page in pages:
                    page.metadata["source"] = filename
                docs.extend(pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        return chunks
    except Exception as e:
        raise RuntimeError(f"Error loading PDFs: {e}")


def get_or_create_vectorstore():
    """
    Load existing ChromaDB if it exists, else build from PDFs.
    This avoids re-embedding on every run.
    """
    embeddings = get_embedding_model()
    try:
        # Check if DB already has data
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        # If collection is empty, build it
        if vectorstore._collection.count() == 0:
            print("ChromaDB is empty. Building embeddings from PDFs...")
            chunks = load_and_split_pdfs()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            vectorstore.persist()
            print(f"Stored {len(chunks)} chunks in ChromaDB.")
        else:
            print(f"Loaded existing ChromaDB with {vectorstore._collection.count()} chunks.")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error with vectorstore: {e}")


def retrieve_context(query: str, vectorstore) -> str:
    """Retrieve top-K relevant chunks for a query."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join([
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        ])
        return context
    except Exception as e:
        return f"Error during retrieval: {e}"