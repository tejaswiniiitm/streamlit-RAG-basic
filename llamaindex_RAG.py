from dotenv import load_dotenv
load_dotenv()
HUGGING_FACE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_documents_get_chunks(docs_dir_path):
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.readers.file import PyMuPDFReader

    file_extractor = {".pdf": PyMuPDFReader()} # to load page numbers as well
    loader = SimpleDirectoryReader(input_dir=docs_dir_path, file_extractor=file_extractor)
    documents = loader.load_data()
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def get_or_create_vector_storage_context(collection_name, vector_db_path):
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext

    db_client = chromadb.PersistentClient(path=vector_db_path)

    chroma_collection = db_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def create_vector_store_db(nodes, vector_db_path, collection_name):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import VectorStoreIndex

    storage_context = get_or_create_vector_storage_context(collection_name, vector_db_path)
    embed_model = HuggingFaceEmbedding(model_name=HUGGING_FACE_MODEL_NAME)
    vector_store_index = VectorStoreIndex(nodes,
                                        storage_context=storage_context,
                                        embed_model=embed_model)
    return vector_store_index

def load_vector_store(vector_db_path, collection_name):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import VectorStoreIndex

    storage_context = get_or_create_vector_storage_context(collection_name, vector_db_path)
    embed_model = HuggingFaceEmbedding(model_name=HUGGING_FACE_MODEL_NAME)
    vector_store_index = VectorStoreIndex.from_vector_store(
                                           vector_store=storage_context.vector_store,
                                          storage_context=storage_context,
                                          embed_model=embed_model)
    return vector_store_index

def get_retriever(vector_store_index):

    from llama_index.core import Settings
    from llama_index.llms.groq import Groq


    Settings.llm = Groq(model="openai/gpt-oss-20b")
    query_engine = vector_store_index.as_query_engine(similarity_top_k=3)
    return query_engine

def document_ingestion(docs_dir_path, vector_db_path, collection_name):
    from utils import Utils
    Utils.download_nltk_punkt()
    text_chunks = load_documents_get_chunks(docs_dir_path)
    print("*************documents loaded")
    create_vector_store_db(text_chunks, vector_db_path, collection_name)
    print ("*************created vector store")

class Llamaindex_RAG_Class:
    @staticmethod
    def ingest_docs_and_get_retriever():
        from utils import Utils
        import os, shutil
        docs_dir_path, vector_db_path = Utils.get_paths()

        vector_db_path += "_llamaindex"
        shutil.rmtree(vector_db_path, ignore_errors=True)
        os.makedirs(vector_db_path)

        collection_name = "llama_doc_collection"
        document_ingestion(docs_dir_path, vector_db_path, collection_name)
        vector_store = load_vector_store(vector_db_path, collection_name)
        query_engine = get_retriever(vector_store)
        return query_engine

    @staticmethod
    def answer_question(retriever, query_text):
        response = retriever.query(query_text)
        answer = response.response
        files_pages = []
        for val in response.metadata.values():
            file_name = val.get("file_name")
            page_label = val.get("source")
            files_pages.append([file_name, page_label])
        return answer, files_pages


def main():
    rag = Llamaindex_RAG_Class
    retriever = rag.ingest_docs_and_get_retriever()
    print("You can now ask question...")
    while(True):
        query_text = input()
        if query_text == "quit":
            break
        answer, files_pages = rag.answer_question(retriever, query_text)
        print(answer)
        print("-" * 50)
        for i in files_pages:
            print(i[0],i[1])

if __name__ == '__main__':
    main()







