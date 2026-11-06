class Utils:
    @staticmethod
    def download_nltk_punkt():
        import nltk
        nltk.download("punkt_tab")
        nltk.download("stopwords")

    @staticmethod
    def get_paths():
        import os
        working_dir = os.path.dirname(os.path.abspath((__file__)))
        docs_dir_path = os.path.join(working_dir, "docs_dir")
        vector_db_path = os.path.join(working_dir, "vector_db")
        return docs_dir_path, vector_db_path