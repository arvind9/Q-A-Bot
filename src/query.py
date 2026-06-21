import os
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client_genai = genai.Client(api_key=api_key)

model_local = SentenceTransformer("all-MiniLM-L6-v2")

def query_rag_pipeline(user_query: str, db_path: str = "./db", k: int = 3) -> dict:
    client_chroma = chromadb.PersistentClient(path=db_path)
    
    collection = client_chroma.get_collection(name="document_knowledge_base")

    query_vector = model_local.encode([user_query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=k
    )

    context_blocks = []
    citations = []

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_name = meta['source']
        page_num = meta['page']
        citation_str = f"Source: {source_name}, Page: {page_num}"

        context_blocks.append(f"[{citation_str}]\nContext: {doc}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    response = client_genai.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": results['documents'][0]
    }