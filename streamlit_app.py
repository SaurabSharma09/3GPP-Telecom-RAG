import os
import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from app.retrieval.retrieval_pipeline import RetrievalPipeline


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="3GPP Telecom RAG",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

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
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Loading Telecom RAG pipeline...")
def load_pipeline():

    pipeline = RetrievalPipeline(
        vector_k=20,
        bm25_k=20,
        hybrid_k=10,
        final_k=5
    )

    return pipeline


@st.cache_resource(show_spinner="Connecting to Groq...")
def load_groq():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# GROUNDED GENERATION
# ============================================================

def generate_answer(
    client,
    query,
    results
):

    if not results:

        return (
            "I could not find sufficient evidence "
            "in the 3GPP documents to answer this question."
        )


    evidence_blocks = []


    for index, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        metadata = chunk.get(
            "metadata",
            {}
        )

        specification = metadata.get(
            "specification",
            "Unknown specification"
        )

        section_number = metadata.get(
            "section_number"
        )

        section_title = metadata.get(
            "section_title",
            "Unknown section"
        )


        if section_number:

            section = (
                f"{section_number} "
                f"{section_title}"
            )

        else:

            section = section_title


        evidence_blocks.append(
            f"""
EVIDENCE {index}

Source:
{specification}

Section:
{section}

Text:
{chunk.get("text", "")}
"""
        )


    evidence_text = "\n".join(
        evidence_blocks
    )


    system_prompt = """
You are a Telecom/3GPP technical expert.

Answer the user's question ONLY using the
retrieved 3GPP evidence provided below.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent section numbers.
4. If the evidence is insufficient, explicitly say:
   "The retrieved 3GPP evidence is insufficient
   to answer this question."
5. Give a clear and technically accurate answer.
6. For procedures, explain the steps in order.
7. For comparisons, clearly separate the responsibilities
   of each network function.
8. Preserve important 3GPP terminology.
9. Keep the answer concise but sufficiently detailed.
10. Cite claims using the source and section supplied
    in the evidence.

Return ONLY the answer.
"""


    user_prompt = f"""
Question:

{query}


Retrieved 3GPP Evidence:

{evidence_text}
"""


    response = client.chat.completions.create(

        model=MODEL_NAME,

        temperature=0,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )


    return response.choices[0].message.content


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def get_sources(results):

    sources = []

    seen = set()


    for result in results:

        chunk = result["chunk"]

        metadata = chunk.get(
            "metadata",
            {}
        )

        specification = metadata.get(
            "specification",
            "Unknown"
        )

        section_number = metadata.get(
            "section_number"
        )

        section_title = metadata.get(
            "section_title",
            "Unknown section"
        )


        key = (
            specification,
            section_number,
            section_title
        )


        if key in seen:

            continue


        seen.add(key)


        sources.append(
            {
                "specification":
                    specification,

                "section_number":
                    section_number,

                "section_title":
                    section_title,

                "text":
                    chunk.get(
                        "text",
                        ""
                    )
            }
        )


    return sources


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📡 3GPP Telecom RAG"
    )

    st.markdown(
        """
        **Retrieval-Augmented Generation
        for 3GPP Telecom Specifications**
        """
    )

    st.divider()

    st.markdown(
        "### Pipeline"
    )

    st.markdown(
        """
        🧠 Query Analysis  
        🔎 FAISS + BM25  
        🔀 Reciprocal Rank Fusion  
        🎯 Cross-Encoder Reranking  
        🛡️ Evidence Gate  
        ✅ Evidence Verification  
        🤖 Groq Generation
        """
    )

    st.divider()

    st.markdown(
        "### Knowledge Base"
    )

    st.write(
        "TS 23.501"
    )

    st.write(
        "TS 23.502"
    )

    st.write(
        "TS 23.503"
    )

    st.divider()

    st.caption(
        "Model: llama-3.3-70b-versatile"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📡 3GPP Telecom RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about 5G architecture, network functions,
    interfaces and procedures using grounded 3GPP evidence.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD BACKEND
# ============================================================

try:

    pipeline = load_pipeline()

    groq_client = load_groq()

except Exception as error:

    st.error(
        f"Failed to initialize the application: {error}"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.markdown(
        "### Try a question"
    )

    examples = [

        "What is the role of the AMF?",

        "What is PDU Session Establishment?",

        "How does PDU Session Establishment work?",

        "What is the role of the SMF?",

        "What is the difference between AMF and SMF?"

    ]


    columns = st.columns(2)


    for index, question in enumerate(
        examples
    ):

        with columns[index % 2]:

            if st.button(
                question,
                use_container_width=True
            ):

                st.session_state.pending_question = (
                    question
                )

                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the 3GPP specifications..."
)


if (
    "pending_question"
    in st.session_state
):

    question = (
        st.session_state.pop(
            "pending_question"
        )
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching 3GPP evidence..."
        ):

            try:

                # ============================================
                # RETRIEVAL PIPELINE
                # ============================================

                output = pipeline.search(
                    question
                )

                results = output.get(
                    "results",
                    []
                )

                evidence = output.get(
                    "evidence",
                    {}
                )

                verification = output.get(
                    "verification",
                    {}
                )


                # ============================================
                # EVIDENCE STATUS
                # ============================================

                if not evidence.get(
                    "sufficient",
                    False
                ):

                    answer = (
                        "The retrieved 3GPP evidence "
                        "is insufficient to answer "
                        "this question."
                    )

                elif verification.get(
                    "status"
                ) != "SUPPORTED":

                    answer = (
                        "The retrieved evidence could "
                        "not be sufficiently verified."
                    )

                else:

                    # ========================================
                    # GROQ
                    # ========================================

                    answer = generate_answer(
                        client=groq_client,
                        query=question,
                        results=results
                    )


                # ============================================
                # ANSWER
                # ============================================

                st.markdown(
                    answer
                )


                # ============================================
                # STATUS
                # ============================================

                st.divider()

                st.markdown(
                    "### 🔎 Evidence"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Gate",
                        (
                            "PASS"
                            if evidence.get(
                                "sufficient"
                            )
                            else "FAIL"
                        )
                    )


                with col2:

                    st.metric(
                        "Verification",
                        verification.get(
                            "status",
                            "UNKNOWN"
                        )
                    )


                with col3:

                    st.metric(
                        "Retrieved",
                        len(results)
                    )


                # ============================================
                # SOURCES
                # ============================================

                sources = get_sources(
                    results
                )


                if sources:

                    st.markdown(
                        "### 📚 Sources"
                    )


                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        if source[
                            "section_number"
                        ]:

                            section = (
                                f"{source['section_number']} "
                                f"{source['section_title']}"
                            )

                        else:

                            section = source[
                                "section_title"
                            ]


                        with st.expander(
                            f"{index}. "
                            f"{source['specification']} — "
                            f"{section}"
                        ):

                            st.write(
                                source["text"]
                            )


                # ============================================
                # SAVE ASSISTANT MESSAGE
                # ============================================

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            answer
                    }
                )


            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            error_message
                    }
                )