from typing import AsyncGenerator
import asyncpg
from asyncpg.pool import Pool
from functools import lru_cache
from app.core.config import settings

# Global pool variable
_pool: Pool | None = None

async def get_pool() -> Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                min_size=2,
                max_size=10
            )
        except Exception as e:
            print(f"Failed to create connection pool: {e}")
            raise
    return _pool

async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get database connection from pool."""
    pool = await get_pool()
    async with pool.acquire() as connection:
        yield connection

# Initialize pool on startup
async def init_db():
    """Initialize database pool."""
    await get_pool()

# Close pool on shutdown    
async def close_db():
    """Close database pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None 