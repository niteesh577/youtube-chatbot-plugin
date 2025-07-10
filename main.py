# main.py - FastAPI backend for YouTube Q&A Assistant

import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("youtube_qa_api")

# Import pipeline
from youtube_utils import get_transcript, get_video_info
from langchain_pipeline import setup_langchain_pipeline, answer_question

# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="YouTube Q&A Assistant API",
    description="Backend for Chrome extension and frontend Q&A interface.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    question: str = Field(..., description="User question about the video")
    timestamp: Optional[str] = Field(None, description="Current video timestamp in MM:SS")

class Citation(BaseModel):
    title: str
    author: str
    url: str
    year: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    timestamp: Optional[str] = None
    citations: List[Citation] = []

# -----------------------------------------------------------------------------
# Exception Handling
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again."}
    )

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Transcript Endpoint
# -----------------------------------------------------------------------------
@app.get("/transcript/{video_id}", tags=["Transcript"])
async def get_video_transcript(video_id: str):
    if not video_id.strip():
        raise HTTPException(status_code=400, detail="Video ID must not be empty.")
    try:
        logger.info(f"🎬 Fetching transcript for video ID: {video_id}")
        transcript = await get_transcript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found.")
        return {"video_id": video_id, "transcript": transcript}
    except Exception as e:
        logger.error(f"Transcript error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching transcript.")

# -----------------------------------------------------------------------------
# Video Info Endpoint
# -----------------------------------------------------------------------------
@app.get("/video/{video_id}", tags=["Video Info"])
async def video_info(video_id: str):
    if not video_id.strip():
        raise HTTPException(status_code=400, detail="Video ID must not be empty.")
    try:
        logger.info(f"ℹ️ Fetching video info for ID: {video_id}")
        info = await get_video_info(video_id)
        return {"video_id": video_id, "info": info}
    except Exception as e:
        logger.error(f"Video info error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching video info.")

# -----------------------------------------------------------------------------
# Chat Endpoint
# -----------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    if not request.video_id.strip():
        raise HTTPException(status_code=400, detail="Video ID must not be empty.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        logger.info(f"💬 Chat request: video={request.video_id}, timestamp={request.timestamp}")
        answer, citations, timestamp = await answer_question(
            request.video_id,
            request.question,
            current_timestamp=request.timestamp
        )

        return ChatResponse(
            answer=answer,
            timestamp=timestamp,
            citations=citations
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error answering your question.")

# -----------------------------------------------------------------------------
# Startup Event
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Bootstrapping LangChain pipeline...")
    try:
        await setup_langchain_pipeline()
        logger.info("✅ LangChain pipeline initialized")
    except Exception as e:
        logger.error(f"❌ Pipeline init failed: {e}")

# -----------------------------------------------------------------------------
# Shutdown Event
# -----------------------------------------------------------------------------
@app.on_event("shutdown")
def shutdown_event():
    logger.info("🛑 API shutdown complete.")

# -----------------------------------------------------------------------------
# Local Dev Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
