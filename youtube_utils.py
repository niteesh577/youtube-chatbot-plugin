# youtube_utils.py - Utilities for working with YouTube videos and transcripts

import os
import re
import logging
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
TranscriptSegment = Dict[str, Any]  # {text, start, duration}
TimestampedTranscript = List[Dict[str, Any]]  # List of {text, timestamp, duration}

# Constants
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")


# ------------------------------------
# Utility to run blocking calls in async context
# ------------------------------------
_executor = ThreadPoolExecutor()

async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


# ------------------------------------
# Get transcript with fallback
# ------------------------------------
def _get_transcript_sync(video_id: str) -> TimestampedTranscript:
    """
    Blocking function to get transcript. Supports fallback to auto-generated.
    """
    languages = ['en', 'en-US', 'hi', 'ja', 'ko', 'kn', 'ml', 'ta', 'te', 'ur', 'vi', 'zh-CN', 'zh-TW']
    try:
        # Attempt normal transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.warning(f"No human transcript for {video_id}: {e}. Trying auto-generated...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id) \
                .find_transcript(languages) \
                .fetch()
        except Exception as inner_e:
            logger.error(f"No transcript (even auto-generated) for {video_id}: {inner_e}")
            raise Exception(
                f"No transcript available for this video in English."
            ) from inner_e
    except Exception as e:
        logger.error(f"Unexpected error retrieving transcript for {video_id}: {e}")
        raise

    # Format the transcript
    formatted_transcript = []
    for segment in transcript_list:
        start_seconds = int(segment['start'])
        minutes = start_seconds // 60
        seconds = start_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        formatted_transcript.append({
            "text": segment['text'],
            "timestamp": timestamp,
            "start": segment['start'],
            "duration": segment['duration']
        })

    return formatted_transcript


async def get_transcript(video_id: str) -> TimestampedTranscript:
    """
    Async wrapper to get YouTube transcript with timestamps.
    """
    return await run_blocking(_get_transcript_sync, video_id)


# ------------------------------------
# Get video info (YouTube API or scraping)
# ------------------------------------
async def get_video_info(video_id: str) -> Dict[str, Any]:
    try:
        if YOUTUBE_API_KEY:
            url = (
                f"https://www.googleapis.com/youtube/v3/videos?"
                f"id={video_id}&key={YOUTUBE_API_KEY}&part=snippet"
            )
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if not data.get('items'):
                raise Exception("Video not found in API response.")

            snippet = data['items'][0]['snippet']
            return {
                "title": snippet.get('title', 'Unknown'),
                "channel": snippet.get('channelTitle', 'Unknown'),
                "description": snippet.get('description', ''),
                "published_at": snippet.get('publishedAt', ''),
                "tags": snippet.get('tags', []),
                "thumbnail": snippet.get('thumbnails', {}).get('high', {}).get('url', '')
            }

        # Fallback to scraping
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('meta', property='og:title')
        channel = soup.find('link', itemprop='name')
        description = soup.find('meta', property='og:description')
        thumbnail = soup.find('meta', property='og:image')

        return {
            "title": title['content'] if title else 'Unknown',
            "channel": channel['content'] if channel else 'Unknown',
            "description": description['content'] if description else '',
            "published_at": '',
            "tags": [],
            "thumbnail": thumbnail['content'] if thumbnail else ''
        }

    except Exception as e:
        logger.error(f"Error getting video info for {video_id}: {e}")
        raise Exception(f"Failed to retrieve video information: {e}")


# ------------------------------------
# Format transcript for context
# ------------------------------------
async def format_transcript_for_context(transcript: TimestampedTranscript) -> str:
    if not transcript:
        return ""

    return "\n".join(
        f"[{seg['timestamp']}] {seg['text']}"
        for seg in transcript
    )


# ------------------------------------
# Find transcript segment by timestamp
# ------------------------------------
async def get_transcript_segment_by_timestamp(
    transcript: TimestampedTranscript,
    timestamp: str
) -> Optional[Dict[str, Any]]:
    if not transcript or not timestamp:
        return None

    match = re.match(r'(\d+):(\d+)', timestamp)
    if not match:
        return None

    minutes, seconds = map(int, match.groups())
    target_seconds = minutes * 60 + seconds

    for segment in transcript:
        start = segment.get('start', 0)
        end = start + segment.get('duration', 0)
        if start <= target_seconds <= end:
            return segment

    closest = min(transcript, key=lambda x: abs(x.get('start', 0) - target_seconds))
    return closest


# ------------------------------------
# Get context around timestamp
# ------------------------------------
async def get_context_around_timestamp(
    transcript: TimestampedTranscript,
    timestamp: str,
    context_window: int = 3
) -> str:
    if not transcript or not timestamp:
        return ""

    target_segment = await get_transcript_segment_by_timestamp(transcript, timestamp)
    if not target_segment:
        return ""

    target_index = next(
        (i for i, seg in enumerate(transcript) if seg.get('start') == target_segment.get('start')),
        -1
    )
    if target_index == -1:
        return ""

    start_index = max(0, target_index - context_window)
    end_index = min(len(transcript), target_index + context_window + 1)

    lines = []
    for i in range(start_index, end_index):
        seg = transcript[i]
        marker = "→ " if i == target_index else "  "
        lines.append(f"{marker}[{seg.get('timestamp', '')}] {seg.get('text', '')}")

    return "\n".join(lines)
