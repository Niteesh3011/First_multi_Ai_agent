import streamlit as st
import time
from pipeline import run_research_pipeline

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="Agentic Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injecting custom CSS for a modern, sleek aesthetic
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem;}
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;}
    .report-container { background-color: #f8fafc; padding: 2rem; border-radius: 12px; border-left: 6px solid #2563EB; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .verdict-box { padding: 1rem; border-radius: 8px; background-color: #EFF6FF; border: 1px solid #BFDBFE; font-weight: 500; color: #1E40AF;}
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #059669; }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER SECTION ---
st.markdown('<p class="main-header">🤖 Autonomous Research Mind</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-agent system powered by LangChain, Mistral AI, and Tavily</p>', unsafe_allow_html=True)

# --- 3. SESSION STATE INITIALIZATION ---
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# --- 4. INPUT SECTION ---
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="Enter a complex topic (e.g., 'Advancements in Agentic AI architectures')",
        label_visibility="collapsed",
    )
with col2:
    execute_btn = st.button("🚀 Initialize Agents", type="primary", use_container_width=True)

# --- 5. DYNAMIC PIPELINE EXECUTION ---
if execute_btn and not topic.strip():
    st.warning("⚠️ Please enter a research topic before launching the agents.")

if execute_btn and topic.strip():
    st.session_state.is_processing = True
    st.session_state.research_data = None  # Reset previous runs

    # Dynamic status container to show real-time agent progression
    with st.status("🧠 Agents are initializing...", expanded=True) as status:
        try:
            st.write("🔍 **Search Agent:** Querying Tavily for live web results...")
            time.sleep(0.3)

            st.write("📖 **Reader Agent:** Scraping and parsing top web pages...")
            # The pipeline runs all 4 stages internally
            results = run_research_pipeline(topic.strip())

            st.write("✍️ **Writer Chain:** Structuring findings into a report...")
            time.sleep(0.3)

            st.write("⚖️ **Critic Chain:** Evaluating report quality and scoring...")
            time.sleep(0.2)

            status.update(label="✅ Research Pipeline Complete!", state="complete", expanded=False)
            st.session_state.research_data = results

        except Exception as e:
            status.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)
            st.error(f"System Error: {e}")
            st.session_state.is_processing = False

# --- 6. RESULTS VISUALIZATION ---
if st.session_state.research_data:
    data = st.session_state.research_data
    feedback = data.get("feedback", {})

    # Top-level metrics from the Critic
    st.divider()
    m1, m2, m3 = st.columns([1, 3, 1])
    with m1:
        score = feedback.get("score", 0)
        st.metric(label="Critic Quality Score", value=f"{score}/10")
    with m2:
        st.markdown(
            f'<div class="verdict-box"><strong>Verdict:</strong> {feedback.get("verdict", "N/A")}</div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.download_button(
            label="📥 Export Report (MD)",
            data=data.get("report", ""),
            file_name=f"{topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.write("")  # Spacing

    # Multi-tab layout for deep data inspection
    tab1, tab2, tab3 = st.tabs(["📄 Final Report", "📊 Critic Analysis", "⚙️ Raw Agent Data"])

    with tab1:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(data.get("report", "No report generated."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        col_s, col_i = st.columns(2)
        with col_s:
            st.subheader("✅ Strengths")
            for strength in feedback.get("strengths", []):
                st.markdown(f"- {strength}")
        with col_i:
            st.subheader("🎯 Areas for Improvement")
            for improvement in feedback.get("improvements", []):
                st.markdown(f"- {improvement}")

    with tab3:
        st.info("This tab displays the raw shared state dictionary passed between the LLM agents.")
        with st.expander("Search Agent Output (Tavily)", expanded=False):
            st.code(data.get("search_results", ""), language="text")
        with st.expander("Reader Agent Output (BeautifulSoup)", expanded=False):
            st.code(data.get("scraped_content", ""), language="text")