"""VeriFind AI - Intelligent Web Information Retrieval Interface.

Streamlit frontend for AMD AI League Match 4.
Accepts natural-language queries, orchestrates navigation, extracts evidence,
and presents verified, citation-backed answers.
"""

import asyncio
import streamlit as st
from backend.answer_engine import AnswerEngine
from backend import VerificationStatus

# Page configuration
st.set_page_config(
    page_title="VeriFind AI - Evidence-Backed Web Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for AMD AI League Branding
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ED1C24 0%, #FF6B4A 50%, #4B79A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-verified {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #c3e6cb;
    }
    .badge-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #ffeeba;
    }
    .badge-unverified {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #f5c6cb;
    }
    .evidence-box {
        border-left: 4px solid #ED1C24;
        background-color: #fcfcfc;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
col_brand, col_badge = st.columns([4, 1])
with col_brand:
    st.markdown('<div class="main-header">🔎 VeriFind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AMD AI League Match 4 · Intelligent Web Information Retrieval Agent</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: #222; color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
                Powered by AMD ROCm & Playwright
            </span>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agent Controls")
    model_provider_choice = st.selectbox(
        "AI Model Provider:",
        options=["gemini", "amd_rocm", "openai_compatible"],
        index=0,
        help="Select between Google Gemini and AMD Developer Cloud ROCm inference backend."
    )
    max_search_results = st.slider("Max Search Results", min_value=1, max_value=8, value=4, step=1)
    timeout_sec = st.slider("Navigation Timeout (sec)", min_value=5, max_value=30, value=15, step=5)
    headless_mode = st.checkbox("Headless Browser Mode", value=True)

    st.markdown("---")
    st.markdown("### 🏆 Match 4 Architecture")
    st.markdown("""
    - **Inference Hardware:** AMD Instinct MI300X / ROCm 6.x
    - **Open-Source Model:** Qwen2.5-7B-Instruct
    - **Dev Provider:** Gemini 3.8 Flash
    - **Browser Automation:** Playwright (Chromium)
    - **Extraction:** Trafilatura & BeautifulSoup4
    """)

# User Input Form
with st.form(key="search_form"):
    query_input = st.text_input(
        "Natural-Language Question:",
        placeholder="e.g. What is the peak memory bandwidth of the AMD Instinct MI300X accelerator?",
        help="Enter the precise fact or question you want the retrieval agent to investigate."
    )

    url_input = st.text_input(
        "Optional Starting URL / Specific Seed Domain:",
        placeholder="e.g. https://www.amd.com or leave blank for open-web search",
        help="Specify an exact page or domain if you want the agent to start navigation from that target."
    )

    submit_button = st.form_submit_button(label="🔍 Search & Verify Evidence", use_container_width=True)

# Main Execution Flow
if submit_button:
    if not query_input or not query_input.strip():
        st.warning("⚠️ Please enter a question or query to start retrieval.")
    else:
        progress_bar = st.progress(0, text="Initializing agent engine...")
        status_box = st.empty()

        async def update_progress(stage: str, fraction: float):
            progress_bar.progress(fraction, text=stage)

        engine = AnswerEngine(
            max_search_results=max_search_results,
            headless=headless_mode,
            timeout_ms=timeout_sec * 1000,
            model_provider=model_provider_choice
        )

        try:
            with st.spinner("Navigating web and extracting evidence..."):
                result = asyncio.run(
                    engine.execute_query(
                        query_str=query_input,
                        starting_url=url_input if url_input.strip() else None,
                        progress_callback=update_progress
                    )
                )

            progress_bar.progress(1.0, text="Retrieval and verification complete!")

            # 1. Answer Section
            st.markdown("### 💡 Direct Answer")
            if result.direct_answer:
                st.info(result.direct_answer)
            else:
                st.warning("⚠️ **Answer: null** (The answer was invalidated because it could not be substantiated by verified evidence or contained contradictions).")

            # Conflicts Warning Callout
            if getattr(result, "conflicts_detected", None):
                st.error("🚨 **Contradictions Detected Across Visited Pages:**")
                for c in result.conflicts_detected:
                    st.markdown(f"- **Claim:** {c.get('claim')} | **Reason:** {c.get('reason')} (Source: {c.get('source_b')})")

            # Invalid Citations Callout
            if getattr(result, "invalid_citations", None):
                with st.expander("⚠️ Rejected / Unverifiable Citations", expanded=False):
                    for inv in result.invalid_citations:
                        st.markdown(f"- {inv}")

            # Extracted Fields & Missing Information callouts
            if result.extracted_fields:
                st.markdown("##### 📌 Extracted Key Fields")
                cols = st.columns(min(4, len(result.extracted_fields)))
                for idx, (field_name, field_val) in enumerate(result.extracted_fields.items()):
                    with cols[idx % len(cols)]:
                        st.metric(field_name.replace("_", " ").title(), str(field_val))

            if result.missing_information:
                st.warning(f"⚠️ **Missing Information:** {result.missing_information}")

            if result.ai_reasoning:
                st.caption(f"🧠 **AI Reasoning:** {result.ai_reasoning}")

            # 2. Verification Status & Confidence
            col_v1, col_v2, col_v3 = st.columns([2, 2, 2])
            with col_v1:
                status_str = result.status.value
                if result.status == VerificationStatus.VERIFIED:
                    badge_html = f'<span class="badge-verified">✓ {status_str}</span>'
                elif result.status == VerificationStatus.PARTIALLY_VERIFIED:
                    badge_html = f'<span class="badge-partial">⚠ {status_str}</span>'
                else:
                    badge_html = f'<span class="badge-unverified">✕ {status_str}</span>'
                st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)

            with col_v2:
                st.metric("Confidence Score", f"{result.confidence_score * 100:.1f}%")

            with col_v3:
                st.metric("Execution Latency", f"{result.execution_time_seconds:.2f}s")

            if result.notes:
                st.caption(f"Verification Rationale: {result.notes}")

            st.divider()

            # 3. Evidence and Citations Section
            st.markdown(f"### 📑 Evidence & Citations ({len(result.evidence)} passages)")
            if result.evidence:
                for idx, ev in enumerate(result.evidence, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="evidence-box">
                            <strong>[Citation #{idx}]</strong> <em>"{ev.snippet}"</em><br/>
                            <small>
                                <strong>Source:</strong> <a href="{ev.source_url}" target="_blank">{ev.source_title or ev.source_url}</a> |
                                <strong>Relevance:</strong> {ev.relevance_score * 100:.1f}% |
                                <strong>Confidence:</strong> {ev.confidence_score * 100:.1f}%
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("No granular evidence passages could be scored for this query.")

            st.divider()

            # 4. Visited Pages Section
            st.markdown(f"### 🌐 Visited Pages ({len(result.visited_pages)})")
            if result.visited_pages:
                table_data = []
                for p in result.visited_pages:
                    table_data.append({
                        "URL": p.url,
                        "Title": p.title,
                        "Status Code": p.status_code,
                        "Content Bytes": p.content_length,
                        "Method": p.extraction_method,
                        "Latency (ms)": p.fetch_latency_ms,
                        "Error": p.error_message or "-"
                    })
                st.dataframe(table_data, use_container_width=True)
            else:
                st.write("No pages were visited during this run.")

        except Exception as e:
            st.error(f"❌ An error occurred during retrieval: {str(e)}")
            st.exception(e)

