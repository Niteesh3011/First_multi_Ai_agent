import re
import logging
import time
from typing import TypedDict

from agents import build_research_agent, build_reader_agent, build_writer_and_critic_chains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Define the shape of our pipeline state explicitly
class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: dict


# --- Helpers ---

_URL_PATTERN = re.compile(r"https?://[^\s\)\]\},\"']+")


def _extract_urls(text: str) -> list[str]:
    """Extract unique URLs from agent output text."""
    urls = _URL_PATTERN.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        # Strip trailing punctuation that regex may capture
        url = url.rstrip(".,;:!?")
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _get_agent_output(result: dict) -> str:
    """Extract the final text answer from a create_agent result.

    The modern create_agent API returns {"messages": [...]}.
    The last message is the final AIMessage with the answer.
    Handles both string content and list-of-blocks content (Mistral format).
    """
    messages = result.get("messages", [])
    if not messages:
        return ""

    content = messages[-1].content

    # Some models (e.g., Mistral) return content as a list of blocks
    # like [{"type": "text", "text": "..."}] instead of a plain string
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(content)


# --- Main Pipeline ---

def run_research_pipeline(topic: str) -> ResearchState:
    """Execute the full research pipeline: Search -> Read -> Write -> Critique."""
    logger.info(f"Starting research pipeline for topic: '{topic}'")
    t_start = time.time()

    # Initialize the state dictionary
    state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": {},
    }

    try:
        # --- 1. RESEARCH AGENT ---
        logger.info("Step 1/4: Executing Search Agent...")
        t1 = time.time()
        research_agent = build_research_agent()
        search_result = research_agent.invoke({
            "messages": [("user", f"Find recent, reliable sources for: {topic}")]
        })
        state["search_results"] = _get_agent_output(search_result)
        logger.info(f"Search Agent completed in {time.time() - t1:.1f}s")

        # --- 2. READER AGENT ---
        logger.info("Step 2/4: Executing Reader Agent...")
        t2 = time.time()

        # Extract URLs programmatically instead of relying on the LLM
        urls = _extract_urls(state["search_results"])
        logger.info(f"Extracted {len(urls)} URLs from search results")

        if urls:
            reader_agent = build_reader_agent()
            scraped_parts: list[str] = []

            # Scrape up to 3 URLs to keep token usage reasonable
            for url in urls[:3]:
                try:
                    logger.info(f"  Scraping: {url}")
                    reader_result = reader_agent.invoke({
                        "messages": [("user", f"Please read and extract information from: {url}")]
                    })
                    output = _get_agent_output(reader_result)
                    if output:
                        scraped_parts.append(f"--- Source: {url} ---\n{output}")
                except Exception as exc:
                    logger.warning(f"  Failed to scrape {url}: {exc}")

            state["scraped_content"] = "\n\n".join(scraped_parts)
        else:
            logger.warning("No URLs found in search results - skipping reader agent")
            state["scraped_content"] = "(No URLs could be extracted for deep reading)"

        logger.info(f"Reader Agent completed in {time.time() - t2:.1f}s")

        # --- 3. WRITER CHAIN ---
        logger.info("Step 3/4: Executing Writer Chain...")
        t3 = time.time()
        writer_chain, critic_chain = build_writer_and_critic_chains()

        combined_research = (
            state["search_results"] + "\n\n" + state["scraped_content"]
        )
        writer_output = writer_chain.invoke({
            "topic": topic,
            "research": combined_research,
        })

        # writer_output is a Pydantic WriterOutput object
        state["report"] = writer_output.report
        logger.info(f"Writer Chain completed in {time.time() - t3:.1f}s")

        # --- 4. CRITIC CHAIN ---
        logger.info("Step 4/4: Executing Critic Chain...")
        t4 = time.time()
        critic_output = critic_chain.invoke({"report": state["report"]})

        # Convert Pydantic model to dict for state storage
        state["feedback"] = critic_output.model_dump()
        logger.info(f"Critic Chain completed in {time.time() - t4:.1f}s")

        total = time.time() - t_start
        logger.info(f"Pipeline completed successfully in {total:.1f}s")
        return state

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise RuntimeError(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    test_topic = input("Enter a research topic:\n> ")
    if not test_topic.strip():
        test_topic = "The current state of Agentic AI frameworks"

    result = run_research_pipeline(test_topic)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["report"])

    print("\n" + "=" * 60)
    print("CRITIC FEEDBACK")
    print("=" * 60)
    feedback = result["feedback"]
    print(f"Score: {feedback.get('score', 'N/A')}/10")
    print(f"Verdict: {feedback.get('verdict', 'N/A')}")
    print(f"Strengths: {feedback.get('strengths', [])}")
    print(f"Improvements: {feedback.get('improvements', [])}")