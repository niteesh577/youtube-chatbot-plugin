# langchain_pipeline.py - LangChain pipeline for YouTube Q&A Assistant with MULTI-TOOL AGENT

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from youtube_utils import (
    get_transcript,
    get_video_info,
    get_context_around_timestamp,
    format_transcript_for_context
)

# -----------------------------------------------------------------------------
# Load environment
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 5

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY is not set!")

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
llm = None
embeddings = None

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
async def setup_langchain_pipeline():
    global llm, embeddings
    if not GEMINI_API_KEY:
        raise EnvironmentError("❌ GEMINI_API_KEY missing in .env")

    logger.info("🔑 Initializing Gemini LLM")
    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        model=MODEL_NAME,
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
    )
    logger.info("💠 Loading embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("✅ LangChain pipeline ready")


# -----------------------------------------------------------------------------
# Create transcript vectorstore
# -----------------------------------------------------------------------------
async def create_transcript_vectorstore(transcript: List[Dict[str, Any]]) -> FAISS:
    if embeddings is None:
        raise ValueError("Embeddings model is not initialized.")
    if not transcript:
        raise ValueError("Transcript is empty.")

    logger.info("🔎 Building vectorstore from transcript")
    documents = [
        Document(
            page_content=segment['text'],
            metadata={"timestamp": segment.get('timestamp', '')}
        )
        for segment in transcript
    ]
    return FAISS.from_documents(documents, embeddings)


# -----------------------------------------------------------------------------
# Research Tool
# -----------------------------------------------------------------------------
async def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        import arxiv
    except ImportError:
        logger.warning("⚠️ arxiv package not installed.")
        return []

    results = []
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        for paper in search.results():
            results.append({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors),
                "summary": paper.summary,
                "url": paper.pdf_url,
                "published": paper.published.strftime("%Y-%m-%d"),
            })
    except Exception as e:
        logger.error(f"arXiv error: {e}")
    return results


async def format_citations(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for paper in papers:
        citations.append({
            "title": paper.get("title", ""),
            "author": paper.get("authors", ""),
            "url": paper.get("url", ""),
            "year": int(paper.get("published", "2000")[:4]) if paper.get("published") else None
        })
    return citations

def clean_research_query(raw_query: str) -> str:
    """
    Remove transcript context or agent system prompts from the query.
    We only want the actual user question for arXiv search.
    """
    if "Current video context" in raw_query:
        return raw_query.split("Current video context")[0].strip()
    return raw_query.strip()



async def create_research_tool() -> Tool:
    async def search_research(query: str) -> str:
        query_clean = clean_research_query(query)
        papers = await search_arxiv(query_clean)
        
        if not papers:
            return "No relevant research papers found."
        
        return "\n\n".join(
            f"{i+1}. {paper['title']} (Authors: {paper['authors']})\n"
            f"URL: {paper['url']}\n"
            f"Summary: {paper['summary'][:200]}..."
            for i, paper in enumerate(papers)
        )


    return Tool(
        name="Research",
        description="Search academic research papers related to the query.",
        func=search_research,
        coroutine=search_research
    )


# -----------------------------------------------------------------------------
# Transcript Semantic Search Tool
# -----------------------------------------------------------------------------
async def create_transcript_search_tool(transcript: List[Dict[str, Any]]) -> Tool:
    vectorstore = await create_transcript_vectorstore(transcript)

    async def search_transcript(query: str) -> str:
        results = vectorstore.similarity_search(query, k=TOP_K_RESULTS)
        if not results:
            return "No relevant information found in the transcript."
        return "\n\n".join(f"[{doc.metadata.get('timestamp', '')}] {doc.page_content}" for doc in results)

    return Tool(
        name="TranscriptSearch",
        description="Search the transcript for relevant passages.",
        func=search_transcript,
        coroutine=search_transcript
    )


# -----------------------------------------------------------------------------
# Timestamp Context Tool
# -----------------------------------------------------------------------------
async def create_timestamp_context_tool(transcript: List[Dict[str, Any]]) -> Tool:
    async def context_tool(query: str) -> str:
        match = re.search(r"(\d{1,2}:\d{2})", query)
        if not match:
            return "Please provide a valid timestamp (MM:SS)."
        timestamp = match.group(1)
        context = await get_context_around_timestamp(transcript, timestamp)
        return context if context else "No context found around that timestamp."
    return Tool(
        name="TimestampContext",
        description="Get transcript text around a specific timestamp.",
        func=context_tool,
        coroutine=context_tool
    )


# -----------------------------------------------------------------------------
# Transcript Global Summary Tool
# -----------------------------------------------------------------------------
async def create_transcript_summary_tool(transcript: List[Dict[str, Any]]) -> Tool:
    full_text = await format_transcript_for_context(transcript)

    logger.info("🧭 Generating summary of full transcript...")
    summary = await llm.apredict(f"Summarize the following video transcript in detail:\n\n{full_text}")

    async def summary_fn(_: str) -> str:
        return summary.strip()

    return Tool(
        name="TranscriptSummary",
        description="Provides a summary of the entire video.",
        func=summary_fn,
        coroutine=summary_fn
    )


# -----------------------------------------------------------------------------
# Fallback QA Tool
# -----------------------------------------------------------------------------
async def create_fallback_qa_tool(transcript: List[Dict[str, Any]]) -> Tool:
    full_text = await format_transcript_for_context(transcript)

    async def fallback_fn(query: str) -> str:
        return await llm.apredict(
            f"Answer the following question based on the entire transcript:\n\nTranscript:\n{full_text}\n\nQuestion:\n{query}"
        )

    return Tool(
        name="FallbackQA",
        description="Fallback tool that answers using the entire transcript if others fail.",
        func=fallback_fn,
        coroutine=fallback_fn
    )


# -----------------------------------------------------------------------------
# Create Agent
# -----------------------------------------------------------------------------
async def create_agent(transcript: List[Dict[str, Any]]) -> AgentExecutor:
    if llm is None or embeddings is None:
        await setup_langchain_pipeline()

    # Build tools
    logger.info("⚙️ Building agent tools")
    transcript_search_tool = await create_transcript_search_tool(transcript)
    timestamp_context_tool = await create_timestamp_context_tool(transcript)
    transcript_summary_tool = await create_transcript_summary_tool(transcript)
    fallback_qa_tool = await create_fallback_qa_tool(transcript)
    research_tool = await create_research_tool()

    tools = [
        transcript_search_tool,
        timestamp_context_tool,
        transcript_summary_tool,
        fallback_qa_tool,
        research_tool
    ]

    # Use LangChain hub ReAct prompt
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        early_stopping_method="generate"
    )


# -----------------------------------------------------------------------------
# Extract timestamps from answer
# -----------------------------------------------------------------------------
async def extract_timestamps_from_answer(answer: str) -> List[str]:
    return re.findall(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', answer)


# -----------------------------------------------------------------------------
# Answer Question (fixed version)
# -----------------------------------------------------------------------------
async def answer_question(
    video_id: str,
    question: str,
    current_timestamp: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    if llm is None or embeddings is None:
        await setup_langchain_pipeline()

    logger.info(f"📹 Fetching transcript for video ID: {video_id}")
    transcript = await get_transcript(video_id)
    video_info = await get_video_info(video_id)

    if not transcript:
        raise ValueError("No transcript available for this video.")

    agent = await create_agent(transcript)

    # Add timestamp context
    if current_timestamp:
        context = await get_context_around_timestamp(transcript, current_timestamp)
        if context:
            question += f"\n\nCurrent video context:\n{context}"

    logger.info(f"🧠 Invoking agent for question: {question}")
    try:
        result = await agent.ainvoke({"input": question})
        answer = result.get("output", "").strip()
    except Exception as e:
        logger.error(f"❌ Agent error: {e}")
        answer = "Sorry, I couldn't process that question."

    try:
        timestamps = await extract_timestamps_from_answer(answer)
        timestamp = timestamps[0] if timestamps else current_timestamp
    except Exception as e:
        logger.warning(f"⚠️ Timestamp extraction failed: {e}")
        timestamp = current_timestamp

    try:
        logger.info(f"🔎 Searching for research citations for: {question}")
        search_query = f"{question} {video_info.get('title', '')}"
        research_results = await search_arxiv(search_query)
        citations = await format_citations(research_results)
    except Exception as e:
        logger.warning(f"⚠️ Citation search failed: {e}")
        citations = []

    return answer or "No answer found.", citations, timestamp
