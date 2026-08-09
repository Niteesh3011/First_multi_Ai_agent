import os
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)

load_dotenv()

# Constants for easy tuning
MAX_SEARCH_RESULTS = 5
SCRAPE_TIMEOUT = 10
MAX_SCRAPE_LINES = 3000

def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is missing from environment variables.")
        raise ValueError("TAVILY_API_KEY is not configured")
    return TavilyClient(api_key=api_key)

@tool
def web_search(query: str) -> str:
    """Perform a web search using Tavily and return concise summaries of top results."""
    logger.info(f"Executing web search for query: '{query}'")
    try:
        client = _get_tavily_client()
        response = client.search(query=query, max_results=MAX_SEARCH_RESULTS)
    except Exception as exc:
        logger.warning(f"Search failed: {exc}")
        return f"Search failed: {exc}"

    if not isinstance(response, dict) or not response.get("results"):
        logger.info("No search results found.")
        return "No results found."

    formatted_results = []
    for result in response["results"][:MAX_SEARCH_RESULTS]:
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content") or result.get("snippet") or ""
        # Clean up whitespace
        content = " ".join(str(content).split())
        formatted_results.append(f"Title: {title}\nURL: {url}\nContent: {content}")

    return "\n------\n".join(formatted_results)

@tool
def scrape_url(url: str) -> str:
    """Scrape the main textual content from a given URL."""
    logger.info(f"Scraping URL: {url}")
    try:
        response = requests.get(
            url, 
            timeout=SCRAPE_TIMEOUT, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"Failed to fetch URL {url}: {exc}")
        return f"Failed to fetch URL: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Strip unnecessary elements to save LLM tokens
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    return "\n".join(lines[:MAX_SCRAPE_LINES])