from typing import AsyncGenerator
import asyncpg
from asyncpg.pool import Pool
from functools import lru_cache
from app.core.config import get_settings
from supabase import create_client


@lru_cache()
def get_supabase():
    """Get cached Supabase client instance."""
    settings = get_settings()
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    ) 