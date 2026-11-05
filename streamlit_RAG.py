import os
import streamlit as st
import shutil

st.title("OpenAI-gpt-oss-20b - Document RAG")

# Initialize session state variables so they persist across reruns
if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "RAG_CLASS" not in st.session_state:
    st.session_state.RAG_CLASS = None

if "saved_files" not in st.session_state:
    st.session_state.saved_files = False



# 1. files uploader widget
uploaded_files_list = st.file_uploader("Upload PDF files",
                                       accept_multiple_files=True,
                                       type=["pdf"],
                                       disabled=st.session_state.saved_files)


# run only once for uploading files
if uploaded_files_list and not st.session_state.saved_files:

    st.success(f"Successfully uploaded {len(uploaded_files_list)} file(s)!")
    # Track successfully saved files
    saved_count = 0

    SAVE_DIR = "docs_dir"
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    os.makedirs(SAVE_DIR)

    VECTOR_DB_DIR = "vector_db"
    shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
    os.makedirs(VECTOR_DB_DIR)

    for uploaded_file in uploaded_files_list:
        # Combine directory path with the original file name
        file_path = os.path.join(SAVE_DIR, uploaded_file.name)

        try:
            # Open a file target in write-bytes ('wb') mode
            with open(file_path, "wb") as f:
                # Write the stream data directly into the file
                f.write(uploaded_file.getbuffer())

            st.write(f"✅ Saved: `{uploaded_file.name}`")
            saved_count += 1

        except Exception as e:
            st.error(f"❌ Failed to save `{uploaded_file.name}`: {e}")

    st.session_state.saved_files = True
    st.rerun() # to allow users to not upload or change uploaded docs, now the uploading files will be disable in next run

if st.session_state.saved_files and st.session_state.RAG_CLASS is None:
    Langchain_option = "Langchain"
    Llamaindex_option = "Llamaindex"
    rag_framework = st.selectbox(
        label = "Which framework would you like to use for RAG?",
        options = (Langchain_option, Llamaindex_option),
        index = None
    )
    if rag_framework is not None:
        st.write("You selected:", rag_framework)
        if (rag_framework == Langchain_option):
            from langchain_RAG import Langchain_RAG_Class
            st.session_state.RAG_CLASS = Langchain_RAG_Class
        elif (rag_framework == Llamaindex_option):
            from llamaindex_RAG import Llamaindex_RAG_Class
            st.session_state.RAG_CLASS = Llamaindex_RAG_Class
        st.rerun() # Rerun to proceed instantly to ingestion with the chosen class

# run only once for docs ingestion in RAG, after files are saved on disk and RAGclass is chosen
if (st.session_state.RAG_CLASS is not None) and (not st.session_state.ingested):
    with st.spinner("Documents ingestion and building RAG is in oven..just wait !"):
        st.session_state.retriever = st.session_state.RAG_CLASS.ingest_docs_and_get_retriever()
        st.session_state.ingested = True


if st.session_state.ingested and st.session_state.retriever is not None:
    st.success("RAG is ready, You can now ask your question")
    user_question = st.text_area("Ask your question about uploaded documents")
    if st.button("Answer"):
        if user_question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                answer, files_pages = st.session_state.RAG_CLASS.answer_question(
                                                st.session_state.retriever,
                                                user_question)
                st.markdown("### OpenAI-gpt-oss-20b")
                st.markdown(answer)
                for f in files_pages:
                    st.markdown(f"**Source:** {f[0]} | **Page:** {f[1]}")


