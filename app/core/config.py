from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
from pydantic import validator

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_API_KEY: str

    # Pinecone Settings
    PINECONE_INDEX: str       # 3072 dimensions for file chunks
    PINECONE_INDEX_TWO: str   # 1536 dimensions for KB articles
    PINECONE_ENVIRONMENT: str

    # Model Settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL_1536: str = "text-embedding-ada-002"
    EMBEDDING_MODEL_3072: str = "text-embedding-3-large"
    TEMPERATURE: float = 0.7

    # API Settings
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    DEBUG: bool = True
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Supabase Settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str | None = None

    # Logging
    LOG_LEVEL: str = "INFO"

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_TOKENS_PER_REQUEST: int = 4000

    # LangChain Settings
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "prex"

    @property
    def allowed_origins_list(self) -> List[str]:
        return self.ALLOWED_ORIGINS.split(",")

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of {allowed_levels}")
        return v.upper()

    @validator("TEMPERATURE")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("TEMPERATURE must be between 0 and 1")
        return v

    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v):
        if not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'")
        return v

    @validator("PINECONE_ENVIRONMENT")
    def validate_pinecone_env(cls, v):
        allowed_envs = ["gcp-starter", "us-west1-gcp-free", "us-east1-gcp", "us-west1-gcp", "us-central1-gcp", "asia-southeast1-gcp"]
        if v not in allowed_envs:
            raise ValueError(f"PINECONE_ENVIRONMENT must be one of {allowed_envs}")
        return v

    class Config:
        env_file = ".env"
        env_file_path = ("backend",)  # Look for .env in backend directory
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 