import os
from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    RESOURCES_PATH: str = "resources"
    DOCUMENTS_PATH = os.path.join(RESOURCES_PATH, "docs")
    CHROMA_PATH = os.path.join(RESOURCES_PATH, "chroma")
    # HUBSPOT_API_KEY: str = os.getenv("HUBSPOT_API_KEY")
    GROQ_API_KEY: str
    GEMINI_API_KEY: str


settings = Settings()