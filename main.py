import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, errors
import pdfplumber
import uuid
from dotenv import load_dotenv
from typing import List
from sklearn.metrics.pairwise import cosine_similarity  # For similarity search
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

CHUNK_SIZE = 200  # Define the chunk size (in words)

# Load environment variables
load_dotenv()

# Configure MongoDB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables.")
client = MongoClient(MONGO_URI)
db = client["pdf_data_db"]
collection = db["pdf_documents"]

# Load Sentence Transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini Pro API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Set up CORS to allow requests from the frontend (localhost:3000)
origins = [
    "http://localhost:3000",  # Frontend URL
    "https://spm-rag-frontend-harsha-senarathnas-projects.vercel.app",  # Vercel deployment URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    question: str

def generate_embedding(text: str) -> List[float]:
    embedding = model.encode(text).tolist()
    return embedding

def cosine_similarity_search(query_embedding, docs_embeddings):
    similarities = cosine_similarity([query_embedding], docs_embeddings)
    return similarities.argsort()[0][::-1]  # Sort by highest similarity

def chunk_text(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with pdfplumber.open(file.file) as pdf:
            # Combine all pages' text
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        # Break text into chunks
        text_chunks = chunk_text(text, CHUNK_SIZE)

        # Store each chunk as a separate document
        chunk_documents = []
        for idx, chunk in enumerate(text_chunks):
            embedding = generate_embedding(chunk)
            chunk_documents.append({
                "filename": file.filename,
                "chunk_id": str(uuid.uuid4()),  # Generate a unique ID for each chunk
                "chunk_index": idx,  # The order of the chunk in the original document
                "content": chunk,
                "embedding": embedding
            })

        # Insert all chunk documents into MongoDB
        collection.insert_many(chunk_documents)

        return {"message": "PDF uploaded and stored in chunks successfully!", "total_chunks": len(chunk_documents)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

@app.post("/ask_question")
async def ask_question(request: PromptRequest):
    try:
        # Generate embedding for the question
        question_embedding = generate_embedding(request.question)

        # Retrieve all documents and their embeddings from MongoDB
        docs = list(collection.find({}, {"content": 1, "embedding": 1}))
        docs_embeddings = [doc["embedding"] for doc in docs]
        
        # Find the most relevant documents using cosine similarity
        relevant_doc_indices = cosine_similarity_search(question_embedding, docs_embeddings)
        
        # Retrieve top matching document(s) based on similarity
        top_docs = " ".join(docs[i]["content"] for i in relevant_doc_indices[:3])  # Get top 3 matches

        # Generate response using Gemini Pro model with the context of top documents
        prompt = f"Question: {request.question}\n\nContext: {top_docs}\n\nAnswer:"
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/check_mongo_connection")
async def check_mongo_connection():
    try:
        # Attempt to ping the MongoDB server
        client.admin.command('ping')
        return {"message": "MongoDB connection is successful!"}
    except errors.ConnectionFailure:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

# Server status message
@app.get("/")
async def root():
    return {"message": "Server is running successfully!"}
