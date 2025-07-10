# langchain_pipeline.py - LangChain pipeline for YouTube Q&A Assistant
import os
import json
from typing import List, Dict, Any, Optional, Tuple

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
# Gemini imports
import google.generativeai as genai
# Local imports

logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_OVERLAP = 200

llm = None

async def setup_langchain_pipeline():
    Initialize the LangChain pipeline with Gemini model and embeddings.
    global llm, embeddings
    # Check if Gemini API key is set
        logger.warning("GEMINI_API_KEY not set. Using default settings.")
    # Configure Gemini
    
    llm = ChatGoogleGenerativeAI(
        temperature=0.3,
        top_k=40,
        safety_settings={
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
    )
    # Initialize the embeddings model
    

async def create_transcript_vectorstore(transcript: List[Dict[str, Any]]) -> FAISS:
    Create a vector store from the transcript for semantic search.
    Args:
        
        A FAISS vector store
    if not transcript:
    
    documents = []
        text = segment['text']
        metadata = {
            "start": segment['start'],
        }
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

    """
    
        query: The search query
        
        A list of paper information
    try:
        
        search = pyarxiv.Search(
            max_results=max_results,
        )
        results = []
            paper_info = {
                "authors": ", ".join(result.authors),
                "published": result.published.strftime("%Y-%m-%d") if result.published else None,
                "arxiv_id": result.get_short_id()
            results.append(paper_info)
        return results
    except Exception as e:
        return []

    """
    
        A LangChain Tool for research
    async def search_research(query: str) -> str:
        if not papers:
        
        for i, paper in enumerate(papers):
            result += f"   Authors: {paper['authors']}\n"
            result += f"   URL: {paper['url']}\n"
        
    
        name="search_research",
        description="Search for research papers related to a query. Use this to find scientific evidence and citations."

async def create_transcript_tool(transcript: List[Dict[str, Any]]) -> Tool:
    Create a tool for searching the video transcript.
    Args:
        
        A LangChain Tool for transcript search
    # Create the vector store
    
        # Search the vector store
        
            return "No relevant information found in the transcript."
        # Format the results
        for i, doc in enumerate(results):
        
    
        name="search_transcript",
        description="Search the video transcript for relevant information. Use this to find what was said in the video."

async def create_agent(transcript: List[Dict[str, Any]], video_info: Dict[str, Any]) -> AgentExecutor:
    Create a LangChain agent with tools for answering questions.
    Args:
        video_info: Information about the video
    Returns:
    """
    transcript_tool = await create_transcript_tool(transcript)
    
    
    prompt = PromptTemplate.from_template(
        
        Title: {video_title}
        
        1. Use the transcript search tool to find relevant parts of the video.
        3. Always include timestamps from the video when referencing content.
        5. Be concise but thorough in your answers.
        7. Format your response in a clear, readable way.
        QUESTION: {question}
        TOOLS:
        
        {agent_scratchpad}
    )
    # Create the agent
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        tools=tools,
        handle_parsing_errors=True,
    )
    return agent_executor

    """
    
        research_results: List of research paper information
    Returns:
    """
    
        citation = {
            "author": paper.get("authors", "Unknown Authors"),
            "year": paper.get("published", "")[:4] if paper.get("published") else None
        citations.append(citation)
    return citations

    """
    
        answer: The answer text
    Returns:
    """
    timestamp_pattern = r'\b(\d{1,2}):(\d{2})\b'
    
    timestamps = [f"{int(m):02d}:{s}" for m, s in matches]
    return timestamps

    """
    
        video_id: The YouTube video ID
        current_timestamp: The current video timestamp (optional)
    Returns:
    """
    
    if llm is None or embeddings is None:
    
    transcript = await get_transcript(video_id)
    
        raise ValueError("No transcript available for this video")
    # Create the agent
    
    input_data = {
        "video_title": video_info.get("title", "Unknown"),
    }
    # Add context from current timestamp if available
        context = await get_context_around_timestamp(transcript, current_timestamp)
            input_data["context"] = f"Current video context:\n{context}"
    # Run the agent
    answer = result.get("output", "")
    # Extract timestamps from the answer
    timestamp = timestamps[0] if timestamps else current_timestamp
    # Search for research papers based on the question and video title
    research_results = await search_arxiv(search_query)
    # Format citations
    