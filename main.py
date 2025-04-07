from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import auth, journal, emotion
from app.api.endpoints.chat import router as chat_router
from app.db import Database
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Access the GitHub token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")

# Create FastAPI app
app = FastAPI(
    title="Emotional Support Diary API",
    description="API for the Emotional Support Diary application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace "*" with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get MongoDB URL from environment variable
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

@app.on_event("startup")
async def startup_db_client():
    """Connect to the database on startup."""
    await Database.connect()

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close the database connection on shutdown."""
    await Database.disconnect()

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(journal.router, prefix="/api/journal", tags=["Journal Entries and Mood Tracker"])
app.include_router(journal.router, tags=["Journal"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Emotional Support Diary API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}