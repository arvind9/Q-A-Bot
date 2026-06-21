import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load a highly popular, fast, local embedding model
model_local = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddingFunction:
    def name(self) -> str:
        return "LocalEmbeddingFunction"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Encodes text into vectors entirely on your computer
        embeddings = model_local.encode(input)
        return embeddings.tolist()

def extract_pdf_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })
            start += (chunk_size - chunk_overlap)
    return chunks

def save_to_vector_db(chunks: list[dict], db_path: str = "./db"):
    client_chroma = chromadb.PersistentClient(path=db_path)
    embedding_fn = LocalEmbeddingFunction()
    
    collection = client_chroma.get_or_create_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    print("Generating embeddings locally (no API needed)...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(chunks)} chunks in the local vector database!")

if __name__ == "__main__":
    data_folder = "./data"
    all_chunks = []
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Created 'data' folder. Drop PDFs in there!")
    else:
        for file in os.listdir(data_folder):
            if file.endswith(".pdf"):
                print(f"Processing {file}...")
                pages = extract_pdf_pages(os.path.join(data_folder, file))
                chunks = chunk_extracted_pages(pages)
                all_chunks.extend(chunks)
        
        if all_chunks:
            save_to_vector_db(all_chunks)
        else:
            print("No PDF files found in the 'data' folder to index.")