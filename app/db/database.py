from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

from app.config import IS_PRODUCTION

_DEFAULT_DB_URL = "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library"
DATABASE_URL = os.getenv("DATABASE_URL", "" if IS_PRODUCTION else _DEFAULT_DB_URL)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set in production.")

# Connection pool tuning — bounded pool with pre-ping so stale connections
# (dropped by the DB/proxy after idle) are transparently recycled.
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=_POOL_SIZE,
    max_overflow=_MAX_OVERFLOW,
    pool_recycle=_POOL_RECYCLE,
)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with async_session_maker() as session:
        yield session

