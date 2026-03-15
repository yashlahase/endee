import os
import io
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from pypdf import PdfReader

# Internal imports
from vector_store.endee_client import EndeeClient
from embeddings.model import EmbeddingModel
from utils.chunking import TextChunker
from rag_pipeline.chain import RAGPipeline

load_dotenv()

app = FastAPI(title="Endee RAG API")

# Initialize components
endee = EndeeClient(
    base_url=os.getenv("ENDEE_URL", "http://localhost:8080"),
    auth_token=os.getenv("ENDEE_AUTH_TOKEN")
)
embeddings = EmbeddingModel(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
chunker = TextChunker(
    chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
)
rag = RAGPipeline()

# Models
class QueryRequest(BaseModel):
    query: str
    index_name: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.get("/health")
def health():
    return {"status": "ok", "endee_connected": endee.health_check()}

@app.post("/upload")
async def upload_document(
    index_name: str = Form(...),
    file: UploadFile = File(...)
):
    """ Uploads a document, chunks it, embeds it, and stores in Endee. """
    
    # 1. Read file
    content = ""
    if file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(await file.read()))
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    else:
        content = (await file.read()).decode("utf-8")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty document content")

    # 2. Chunk text
    chunks = chunker.split_text(content)
    
    # 3. Generate embeddings
    vectors = embeddings.generate(chunks)
    
    # 4. Ensure index exists
    existing_indexes = endee.list_indexes()
    index_names = [idx["name"].split("/")[-1] for idx in existing_indexes] # Endee uses username/index_name
    
    if index_name not in index_names:
        success = endee.create_index(index_name, dimension=embeddings.dimension)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create Endee index")

    # 5. Prepare vectors for Endee
    # Endee HybridVectorObject format: {id, vector, meta, filter, norm}
    payload = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload.append({
            "id": f"{file.filename}_{uuid.uuid4().hex[:8]}",
            "vector": vec,
            "meta": chunk,  # Storing the original text in metadata for RAG
            "filter": f"source:{file.filename}"
        })

    # 6. Insert into Endee
    success = endee.insert_vectors(index_name, payload)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to insert vectors into Endee")

    return {"message": "Document indexed successfully", "chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """ Performs semantic search and generates an answer using LLM. """
    
    # 1. Generate query embedding
    query_vec = embeddings.generate_single(request.query)
    
    # 2. Search Endee
    hits = endee.search(request.index_name, query_vec, k=request.k)
    
    if not hits:
        return QueryResponse(answer="No relevant context found in the database.", sources=[])

    # 3. Run RAG pipeline
    answer = rag.run(request.query, hits)
    
    return QueryResponse(answer=answer, sources=hits)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
