import streamlit as st
from parser import extract_text_from_pdf
from pipeline import RAGPipeline
from model import LocalGPTModel
import tempfile
from streamlit_pdf_viewer import pdf_viewer
import time

def run_app():
    st.set_page_config(page_title=" Research Assistant", layout="wide")

    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "llm" not in st.session_state:
        st.session_state.llm = None

    # Upload Screen
    if not st.session_state.pdf_uploaded:
        st.markdown("<div style='text-align: center; margin-top: 150px;'>", unsafe_allow_html=True)
        st.title("Research Assistant for First-Time Readers")
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf", label_visibility="visible")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.pdf_path = tmp_file.name

            with st.spinner("🔍 Processing and indexing the paper..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                st.session_state.rag = RAGPipeline(raw_text)
                st.session_state.llm = LocalGPTModel()
                time.sleep(2)

            st.session_state.pdf_uploaded = True
            st.rerun()

    # PDF and Chat Interface
    else:
        st.title("Explore Your Paper")
        pdf_col, chat_col = st.columns([2, 1])

        with pdf_col:
            st.subheader(" Paper Viewer")
            pdf_viewer(st.session_state.pdf_path, width=700, height=1000)

        with chat_col:
            st.subheader(" Ask a Question")
            with st.container():
                st.markdown(
                    """
                    <div style='background-color:#1e1e1e; padding: 20px; border-radius: 12px;'>
                    """, unsafe_allow_html=True)

                user_question = st.text_input("", placeholder="Ask your question here...")

                if user_question:
                    with st.spinner(" Thinking..."):
                        chunks, scores = st.session_state.rag.retrieve_relevant_chunks(user_question)
                        top_score = scores[0]

                        if top_score <= 0.3:
                            response = (
                                " The paper doesn’t directly cover this. "
                                "I’ll explore online to help find an answer."
                            )
                        else:
                            context = " ".join(chunks)
                            response = st.session_state.llm.generate_answer(user_question, context)

                    st.markdown("###  Answer")
                    st.success(response)

                    if top_score > 0.3:
                        with st.expander("Context from paper"):
                            for i, chunk in enumerate(chunks):
                                st.markdown(f"**Chunk {i+1}**: {chunk[:500]}...")

                st.markdown("</div>", unsafe_allow_html=True)
