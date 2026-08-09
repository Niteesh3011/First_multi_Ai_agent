import os
import logging
import functools
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from tools import web_search, scrape_url

logger = logging.getLogger(__name__)
load_dotenv()


# --- Pydantic Data Models for Structured Output ---
class WriterOutput(BaseModel):
    report: str = Field(description="The detailed, structured research report.")
    sources: list[str] = Field(description="A list of all URLs referenced in the report.")


class CriticOutput(BaseModel):
    score: int = Field(description="A score out of 10 for the report quality.")
    strengths: list[str] = Field(description="List of the report's strengths.")
    improvements: list[str] = Field(description="List of areas to improve.")
    verdict: str = Field(description="A one-line summary verdict.")


@functools.lru_cache(maxsize=1)
def _get_llm() -> ChatMistralAI:
    """Return a cached, production-ready LangChain-compatible Mistral chat LLM.

    Uses lru_cache to avoid creating multiple identical instances per pipeline run.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY is not configured")
        raise RuntimeError("MISTRAL_API_KEY is not configured")

    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=0.2,
        api_key=api_key,
    )


### 1. RESEARCH AGENT (using modern create_agent API)
def build_research_agent():
    """Build a LangGraph-based agent that searches the web for research material.

    Returns a CompiledStateGraph. Invoke with:
        result = agent.invoke({"messages": [("user", "...")]})
        answer = result["messages"][-1].content
    """
    agent = create_agent(
        model=_get_llm(),
        tools=[web_search, scrape_url],
        system_prompt=(
            "You are an expert research assistant. Search the web, scrape "
            "relevant content, and provide a comprehensive summary of your "
            "findings. Always include the source URLs in your output."
        ),
    )
    return agent


### 2. READER AGENT
def build_reader_agent():
    """Build a LangGraph-based agent that scrapes URLs and extracts content.

    Returns a CompiledStateGraph. Invoke with:
        result = agent.invoke({"messages": [("user", "...")]})
        answer = result["messages"][-1].content
    """
    agent = create_agent(
        model=_get_llm(),
        tools=[scrape_url],
        system_prompt=(
            "You are an expert data extractor. Scrape the provided URL(s) and "
            "summarize the core factual content from each page."
        ),
    )
    return agent


### 3. WRITER & CRITIC CHAINS (LCEL pipelines with structured output)
def build_writer_and_critic_chains() -> tuple[object, object]:
    """Build the writer and critic LCEL chains with structured Pydantic output."""
    llm = _get_llm()

    # --- Writer Chain ---
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert research writer. Write clear, structured, and "
         "insightful reports based ONLY on the provided research."),
        ("human", """Topic: {topic}

Research Gathered:
{research}

Structure the report with an Introduction, Key Findings (minimum 3), and a Conclusion."""),
    ])

    writer_chain = writer_prompt | llm.with_structured_output(WriterOutput)

    # --- Critic Chain ---
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a ruthless but highly constructive senior editor."),
        ("human",
         "Review this research report and provide a strict evaluation:\n\n{report}"),
    ])

    critic_chain = critic_prompt | llm.with_structured_output(CriticOutput)

    return writer_chain, critic_chain