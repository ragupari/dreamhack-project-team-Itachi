from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings

_db_instance = None

def get():
    global _db_instance
    if _db_instance is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _db_instance = Chroma(
            persist_directory=str(settings.CHROMA_PATH),
            embedding_function=embeddings
        )
    return _db_instance