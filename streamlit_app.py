import streamlit as st
from dotenv import load_dotenv

from app.retrieval.retrieval_pipeline import RetrievalPipeline


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="3GPP Telecom RAG",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        background-color: #fafafa;
    }

    .source-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .metric-card {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: #fafafa;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# LOAD PIPELINE
# ============================================================================


@st.cache_resource(show_spinner="Loading Telecom RAG pipeline...")
def load_pipeline():

    return RetrievalPipeline(
        vector_k=40,
        bm25_k=40,
        hybrid_k=20,
        final_k=15,
    )


# ============================================================================
# SOURCE HELPERS
# ============================================================================


def get_sources(results, limit=5):
    """
    Build a clean deduplicated source list.

    Only the strongest few retrieved sources are shown in the UI.
    """

    sources = []
    seen = set()

    for result in results[:limit]:

        chunk = result.get("chunk", {})

        metadata = chunk.get("metadata", {})

        specification = metadata.get("specification", "Unknown")

        section_number = metadata.get("section_number")

        section_title = metadata.get("section_title", "Unknown section")

        section_type = metadata.get("section_type")

        # Do not prominently display cover/document metadata as
        # technical evidence.
        if section_type == "cover":
            continue

        key = (
            specification,
            section_number,
            section_title,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "specification": specification,
                "section_number": section_number,
                "section_title": section_title,
                "text": chunk.get("text", ""),
                "score": result.get("combined_rerank_score", 0.0),
            }
        )

    return sources


# ============================================================================
# FORMAT SOURCE TITLE
# ============================================================================


def format_section(source):
    section_number = source.get("section_number")

    section_title = source.get("section_title", "Unknown section")

    if section_number:
        return f"{section_number} " f"{section_title}"

    return section_title


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("## 📡 3GPP Telecom RAG")

    st.markdown(
        """
        **Retrieval-Augmented Generation
        for 3GPP Telecom Specifications**
        """
    )

    st.divider()

    st.markdown("### Pipeline")

    st.markdown(
        """
        🧠 Query Analysis  
        🔎 FAISS + BM25  
        🔀 Reciprocal Rank Fusion  
        🎯 Cross-Encoder Reranking  
        🛡️ Evidence Gate  
        ✅ Lexical Verification  
        🧠 Semantic Verification  
        🤖 Grounded Generation
        """
    )

    st.divider()

    st.markdown("### Knowledge Base")

    st.write("TS 23.501")

    st.write("TS 23.502")

    st.write("TS 23.503")

    st.divider()

    st.caption("Generation / verification model: " "openai/gpt-oss-120b")


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    '<div class="main-title">📡 3GPP Telecom RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about 5G architecture, network functions,
    interfaces and procedures using grounded 3GPP evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# INITIALIZE BACKEND
# ============================================================================

try:

    pipeline = load_pipeline()

except Exception as error:

    st.error(f"Failed to initialize the application: {error}")

    st.stop()


# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================================
# EXAMPLE QUESTIONS
# ============================================================================

if not st.session_state.messages:

    st.markdown("### Try a question")

    examples = [
        "What is the role of the AMF?",
        "What is the role of the SMF?",
        "What is the role of the UPF?",
        "What is the N4 interface?",
        "How does UE registration work in the 5G system?",
        "What is PDU Session Establishment?",
        "What is the difference between AMF and SMF?",
        "What is the capital of France?",
    ]

    columns = st.columns(2)

    for index, question in enumerate(examples):

        with columns[index % 2]:

            if st.button(
                question,
                use_container_width=True,
                key=f"example_{index}",
            ):

                st.session_state["pending_question"] = question

                st.rerun()


# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("details"):

            details = message["details"]

            st.divider()

            st.markdown("#### 🔎 Verification")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric("Gate", details.get("gate", "UNKNOWN"))

            with col2:

                st.metric("Lexical", details.get("lexical", "UNKNOWN"))

            with col3:

                st.metric("Semantic", details.get("semantic", "UNKNOWN"))

            with col4:

                st.metric("Evidence", details.get("evidence_count", 0))


# ============================================================================
# INPUT
# ============================================================================

question = st.chat_input("Ask a question about the 3GPP specifications...")


# ============================================================================
# EXAMPLE BUTTON INPUT
# ============================================================================

if "pending_question" in st.session_state:

    question = st.session_state.pop("pending_question")


# ============================================================================
# PROCESS QUESTION
# ============================================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # ------------------------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ------------------------------------------------------------------------
    # ASSISTANT
    # ------------------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching and verifying 3GPP evidence..."):

            try:

                # ============================================================
                # CANONICAL PIPELINE
                # ============================================================

                output = pipeline.search(question)

                # ============================================================
                # EXTRACT PIPELINE OUTPUT
                # ============================================================

                answer = output.get(
                    "answer",
                    (
                        "The retrieved 3GPP evidence is "
                        "insufficient to answer this question."
                    ),
                )

                results = output.get("results", [])

                evidence = output.get("evidence", {})

                verification = output.get("verification", {})

                semantic = output.get("semantic_verification", {})

                final_status = output.get("final_status", "NOT_SUPPORTED")

                grounding = output.get("grounding", {})

                # ============================================================
                # ANSWER
                # ============================================================

                st.markdown(answer)

                # ============================================================
                # STATUS
                # ============================================================

                st.divider()

                st.markdown("### 🔎 Verification")

                col1, col2, col3, col4 = st.columns(4)

                # ------------------------------------------------------------
                # Gate
                # ------------------------------------------------------------

                with col1:

                    gate_status = (
                        "PASS" if evidence.get("sufficient", False) else "FAIL"
                    )

                    st.metric("Evidence Gate", gate_status)

                # ------------------------------------------------------------
                # Lexical
                # ------------------------------------------------------------

                with col2:

                    st.metric("Lexical", verification.get("status", "UNKNOWN"))

                # ------------------------------------------------------------
                # Semantic
                # ------------------------------------------------------------

                with col3:

                    st.metric("Semantic", semantic.get("verdict", "UNKNOWN"))

                # ------------------------------------------------------------
                # Evidence
                # ------------------------------------------------------------

                with col4:

                    st.metric("Retrieved", len(results))

                # ============================================================
                # FINAL STATUS
                # ============================================================

                st.caption(
                    f"Final status: {final_status} | "
                    f"Grounding: "
                    f"{grounding.get('status', 'UNKNOWN')}"
                )

                # ============================================================
                # VERIFICATION DETAILS
                # ============================================================

                with st.expander("View verification details"):

                    st.write(
                        {
                            "Evidence Gate": evidence,
                            "Lexical Verification": verification,
                            "Semantic Verification": semantic,
                        }
                    )

                # ============================================================
                # SOURCES
                # ============================================================

                sources = get_sources(results, limit=5)

                if sources:

                    st.markdown("### 📚 Sources")

                    for index, source in enumerate(sources, start=1):

                        section = format_section(source)

                        with st.expander(
                            f"{index}. " f"{source['specification']} — " f"{section}"
                        ):

                            st.caption("Rerank score: " f"{source['score']:.4f}")

                            st.write(source["text"])

                # ============================================================
                # EVIDENCE SUMMARY
                # ============================================================

                st.markdown("### 📊 Grounding Summary")

                summary_col1, summary_col2 = st.columns(2)

                with summary_col1:

                    st.write(
                        f"**Evidence chunks:** "
                        f"{grounding.get('evidence_chunks', 0)}"
                    )

                    st.write(
                        f"**Unique sources:** " f"{grounding.get('source_count', 0)}"
                    )

                with summary_col2:

                    st.write(
                        f"**Semantic confidence:** "
                        f"{semantic.get('confidence', 0.0):.2f}"
                    )

                    st.write(f"**Reason:** " f"{semantic.get('reason', '')}")

                # ============================================================
                # SAVE MESSAGE
                # ============================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "details": {
                            "gate": gate_status,
                            "lexical": verification.get("status", "UNKNOWN"),
                            "semantic": semantic.get("verdict", "UNKNOWN"),
                            "evidence_count": len(results),
                        },
                    }
                )

            except Exception as error:

                error_message = f"Something went wrong: {error}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
