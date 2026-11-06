
from dotenv import load_dotenv
load_dotenv()
HUGGING_FACE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Download nd Install Poppler and Tesseract OCR - to convert PDF pages into images and extract their text
# Since Windows doesn't have a native package manager like Mac or Linux,
# you must download pre-compiled binaries and add them to your system's env Variables manually.


def load_documents_get_chunks(docs_dir_path):
    from langchain_unstructured.document_loaders import UnstructuredLoader
    from pathlib import Path

    pdf_files = list(Path(docs_dir_path).glob("**/*.pdf"))
    loader = UnstructuredLoader(file_path=pdf_files, chunking_strategy="by_title")
    documents = loader.load()

    # initializing the text splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def create_vector_store_db(text_chunks, vector_db_path, collection_name):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embedding_model = HuggingFaceEmbeddings(model_name=HUGGING_FACE_MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        persist_directory=vector_db_path,
        collection_name=collection_name
    )
    print("created vector store")

def load_vector_store(vector_db_path, collection_name):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embedding_model = HuggingFaceEmbeddings(model_name=HUGGING_FACE_MODEL_NAME)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=vector_db_path
    )
    return vector_store

def get_retriever(vector_store):
    from langchain_groq import ChatGroq
    from langchain_classic.chains import RetrievalQA

    retriever = vector_store.as_retriever()
    llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def document_ingestion(docs_dir_path, vector_db_path, collection_name):
    from utils import Utils
    Utils.download_nltk_punkt()
    text_chunks = load_documents_get_chunks(docs_dir_path)
    create_vector_store_db(text_chunks, vector_db_path, collection_name)

class Langchain_RAG_Class:
    @staticmethod
    def ingest_docs_and_get_retriever():
        from utils import Utils
        import os, shutil

        docs_dir_path, vector_db_path = Utils.get_paths()
        vector_db_path += "_langchain"
        shutil.rmtree(vector_db_path, ignore_errors=True)
        os.makedirs(vector_db_path)

        collection_name = "langchain_doc_collection"
        document_ingestion(docs_dir_path, vector_db_path, collection_name)
        vector_store = load_vector_store(vector_db_path, collection_name)
        retrieval_chain = get_retriever(vector_store)
        return retrieval_chain

    @staticmethod
    def answer_question(retriever, query_text):
        response = retriever.invoke({"query": query_text})
        answer = response["result"]
        files_pages = []
        for source in response["source_documents"]:
            doc_name = source.metadata['source'].split('\\')[-1]
            pg_num = source.metadata['page_number']
            files_pages.append([doc_name, pg_num])
        return answer, files_pages


def main():
    rag = Langchain_RAG_Class
    retrieval_chain = rag.ingest_docs_and_get_retriever()
    while(True):
        query_text = input()
        if query_text == "quit":
            break
        answer, files_pages = rag.answer_question(retrieval_chain, query_text)
        print(answer)
        for i in files_pages:
            print(i[0], i[1])


if __name__ == '__main__':
    main()





